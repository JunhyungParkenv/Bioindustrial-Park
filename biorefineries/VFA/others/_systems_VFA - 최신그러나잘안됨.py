# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:30:14 2024

@author: Junhyung Park
"""

# %% Setup

import biosteam as bst
import thermosteam as tmo
from biosteam import Stream, SystemFactory
from biosteam.process_tools import SystemFactory
from biosteam import main_flowsheet
from biorefineries.cellulosic import units
from biorefineries.VFA import _chemicals
from biorefineries.VFA import _units
from biorefineries.VFA._chemicals import chems, chemical_groups, get_grouped_chemicals
from biorefineries.VFA._process_settings import load_preferences_and_process_settings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

load_preferences_and_process_settings()  # Flow 단위를 'kg/hr'로 설정

# Thermodynamic properties
tmo.settings.set_thermo(chems)

# ✅ **🔹 Global Variable for Target Concentration**
target_concentration = 500  # g/L

# Flowsheet Initialization
F = bst.Flowsheet('VFA_Recovery')
bst.main_flowsheet.set_flowsheet(F)
# %% System Definition
# 시스템 모듈에서 target_concentration을 설정하여 ED 및 Tank 조정
@SystemFactory(
    ID='VFA_sys',
    ins=[dict(ID='feedstock', units='kg/hr')],
    outs=[
        dict(ID='stored_vfa', units='kg/hr'), 
        dict(ID='dc_output', units='kg/hr'),
        dict(ID='biogas', units='kg/hr'),
        dict(ID='U302_cell_mass', units='kg/hr')
    ]
)

def create_VFA_sys(ins, outs):
    """VFA Recovery System: Anaerobic digestion and Electrodialysis-based separation"""
    feedstock = ins[0]
    stored_vfa, dc_output, biogas, U302_cell_mass = outs

    # --- Feedstock Initialization ---
    feedstock.imass['Water'] = 154570.91
    feedstock.imass['Glucose'] = 3154.51
    feedstock.price = 0.1  # Price per kg

    # --- 1. Anaerobic Digestion (UASB Reactor) ---
    R101 = _units.UASB('R101', ins=feedstock, outs=('vfa_solution', biogas))

    # --- 2. Solid-Liquid Separation ---
    U302 = _units.CellMassFilter(
        'U302',
        ins=R101-0,  # vfa_solution
        outs=(U302_cell_mass, 'vfa_filtered'),
        moisture_content=None,
        split=0.0
    )

    # --- 3. Split into inf_dc and inf_ac ---
    S302 = bst.Splitter(
        'S302',
        ins=U302-1,  # vfa_filtered
        outs=('fresh_dc', 'fresh_ac'),
        split=0.8  # 80% inf_dc, 20% inf_ac
    )

    # --- 5. Recycle Streams ---
    recycle_dc = bst.Stream('recycle_dc')
    recycle_ac = bst.Stream('recycle_ac')

    # --- 6. Mix Tanks (DC Tank, AC Tank) ---
    T301 = _units.DC_Tank(
        'dc_tank',
        ins=(S302-0, recycle_dc),
        outs='tank_to_dc',
        tau=24  # 기본값, 나중에 업데이트될 예정
    )
    T302 = _units.AC_Tank(
        'ac_tank',
        ins=(S302-1, recycle_ac),
        outs='tank_to_ac',
        tau=6  # 나중에 업데이트됨
    )

    # --- 7. Electrodialysis (ED) ---
    S401 = _units.ED(
        'S401',
        ins=(T301-0, T302-0),
        outs=('treated_dc', 'treated_ac'),
        j=110.375,
        t=240*3600,
        A_m=1.0,  # 초기 멤브레인 면적, 이후 업데이트됨
        target_concentration=target_concentration
    )
    
    @S401.add_specification(run=True)
    def update_ed_parameters():
        """ED 유닛과 AC Tank 설정 업데이트"""
        eff_ac = S401.outs[1]
        total_vfa_mass = eff_ac.imass['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid'].sum()  # kg/hr
        total_vfa_mol = total_vfa_mass / 60.05  # kmol/hr (평균 분자량 60.05 g/mol)
        
        I = S401.j * S401.A_m  # 총 전류
        flux_dict = S401.calculate_flux(I)
        total_flux = sum(flux_dict.values())  # mol/(m²·s)
    
        # ✅ 목표 농도 반영한 A_m 조정
        Q = eff_ac.F_vol  # m³/hr
        S401.A_m = S401.calculate_membrane_area(total_vfa_mol, total_flux, Q)
    
        print(f"🔹 Updated ED Membrane Area: {S401.A_m:.4f} m²")
        
        # ✅ AC Tank 체류 시간 업데이트
        update_ac_tau_based_on_target_concentration()

    
    

    # def update_ac_tau_based_on_target_concentration():
    #     """목표 농도를 반영한 AC Tank 체류시간 (tau) 조정"""
    #     T302._design()  # AC Tank의 design_results 강제 업데이트
        
    #     # ✅ 목표 농도 (mol/m³)
    #     C_target_ac = (target_concentration * 10 / 60.05) / 1000  # g/L → mol/m³ 변환
    
    #     # AC Tank 부피 (m³)
    #     V_ac = T302.design_results['Volume']
    
    #     # 🔹 AC Tank의 현재 총 VFA 질량 (kg/hr)
    #     total_vfa_mass_ac = T302.outs[0].imass['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid'].sum()
    
    #     # 🔹 AC Tank의 현재 용적 유량 (m³/hr)
    #     total_solution_volume_ac = T302.outs[0].F_vol  # m³/hr
    
    #     # ✅ 현재 AC Tank 출력 농도 (g/L)
    #     if total_solution_volume_ac > 1e-6:
    #         current_concentration_ac = (total_vfa_mass_ac / total_solution_volume_ac)  # kg/m³ = g/L
    #     else:
    #         print("⚠ Warning: AC Tank volume is too low, skipping tau adjustment.")
    #         return
    
    #     # ✅ 목표 농도와 비교하여 tau 조정
    #     if current_concentration_ac < target_concentration:  
    #         T302.tau = (V_ac * C_target_ac) / (total_solution_volume_ac + 1e-6)
        
    #     print(f"✅ Updated AC Tank tau: {T302.tau:.4f} hr")
    def update_ac_tau_based_on_target_concentration():
        """목표 농도를 반영한 AC Tank 체류시간 (tau) 조정"""
        T302._design()  # AC Tank의 design_results 강제 업데이트
    
        # ✅ 목표 농도 (kg/m³)
        C_target_ac = target_concentration * 10 # g/L → kg/m³ 변환
    
        # ✅ AC Tank 부피 (m³)
        V_ac = T302.design_results['Volume']
    
        # 🔹 AC Tank로 유입되는 총 VFA 질량 (kg/hr)
        total_vfa_mass_ac = S_AC.outs[0].imass['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid'].sum()
    
        # 🔹 AC Tank로 유입되는 유량 (m³/hr)
        Q_in_ac = S_AC.outs[0].F_vol  # m³/hr
    
        # ✅ 목표 농도를 달성하기 위해 필요한 배출 유량 (m³/hr)
        if C_target_ac > 1e-6:
            Q_out_ac = total_vfa_mass_ac / C_target_ac  # m³/hr
        else:
            print("⚠ Warning: Target concentration is too low, skipping tau adjustment.")
            return
    
        # ✅ 체류 시간 (tau) 계산
        if Q_out_ac > 1e-6:
            T302.tau = V_ac / Q_out_ac  # hr
        else:
            print("⚠ Warning: AC Tank outflow is too low, skipping tau adjustment.")
            return
    
        print(f"✅ Updated AC Tank tau: {T302.tau:.4f} hr")





    # --- 6. DC Output Handling (재순환 포함) ---
    S_DC = bst.Splitter(
        'S_DC',
        ins=S401-0,  # ED의 DC 출력
        outs=(recycle_dc, dc_output),
        split=0.5 # 50% 재순환, 50% 배출
    )
    
    # --- 8. AC Output Handling (재순환 포함) ---
    S_AC = bst.Splitter(
        'S_AC',
        ins=S401-1,  # ED의 AC 출력
        outs=(recycle_ac, 'ac_for_MEE'),
        split=0.5  # 10% 재순환, 90% MEE로 이동
    )

    # # --- 9. Multi-Effect Evaporator (MEE) ---
    # E101 = bst.MultiEffectEvaporator(
    #     'E101',
    #     ins=S_AC-1,  # `ac_for_MEE`가 4000 g/L 농도를 만족해야 함
    #     outs=('vfa_evaporated', evaporated_water),
    #     V=0,
    #     P=(101325, 73581, 50892, 32777, 20000)
    # )

    # --- 5. Crystallization ---
    S201 = bst.BatchCrystallizer(
        'S201',
        ins=S_AC-1,
        outs='solid_vfa',
        tau=6,  # Residence time
        N=4,  # Number of crystallizers
        T=273.15+0.25  # Temperature
    )

    # --- 6. 추가적인 Drying (선택 가능) ---
    D301 = bst.DrumDryer(
        'D301',
        ins=S201-0,  # Crystallizer output
        outs='dried_vfa',
        moisture_content=0.05,  # 최종 수분 함량 5% 목표
        split={'Water': 0.95},  # 물 95% 제거
        T=343.15  # 건조 온도 (섭씨 70도)
    )
    
    # --- 7. Storage ---
    T101 = bst.StorageTank(
        'T101',
        ins=D301-0,  # Dryer output
        outs=stored_vfa,
        tau=7*24  # 7-day storage time, similar to ethanol's in Humbird et al. 
    )

    # Return all units for inspection (optional)
    # return [R101, U302, S401, E101, S201, T101]

#%%
# VFA System
VFA_sys = create_VFA_sys()
VFA_sys.diagram('cluster', number=True, format='png')
#%%
VFA_sys.simulate()
VFA_sys.show()
#%%
# DC/AC Tank의 체류 시간과 ED 유닛의 디자인 결과 출력
dc_tank = F.unit['dc_tank']
ac_tank = F.unit['ac_tank']
ed_unit = F.unit['S401']

# 결과 출력
print("--- DC/AC Tank and ED Design Information ---")
print(f"DC Tank Residence Time (tau): {dc_tank.tau} hr, Total Volume: {dc_tank.design_results['Total volume']:.4f} m³")
print(f"AC Tank Residence Time (tau): {ac_tank.tau} hr, Total Volume: {ac_tank.design_results['Total volume']:.4f} m³")
print(f"ED Required Membrane Area (A_m): {ed_unit.design_results['Membrane area']:.4f} m²")
print(f"ED Adjusted Current Density (j): {ed_unit.j:.4f} A/m²")
print(f"ED Power Consumption: {ed_unit.design_results['Power consumption']:.4f} W")