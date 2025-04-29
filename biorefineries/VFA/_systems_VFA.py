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
# target_concentration = 2.694  # g/L # 0.898 (ED -> DC) * 3 -> 
target_concentration = 15  # g/L # 0.898 (ED -> DC) * 3 -> / 14.05 g/L (F.T302.outs[0]), 1.42 g/L (F.S401.ins[0])
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
        'T301',
        ins=(S302-0, recycle_dc),
        outs='tank_to_dc',
        tau=6  # 기본값, 나중에 업데이트될 예정
    )
    T302 = _units.AC_Tank(
        'T302',
        ins=(S302-1, recycle_ac),
        outs='tank_to_ac',
        tau=6  # 나중에 업데이트됨
    )

    # --- 7. Electrodialysis (ED) ---
    S401 = _units.ED(
        'S401',
        ins=(T301-0, T302-0),
        outs=('treated_dc', 'treated_ac'),
        j=12.5, # 12.5 (실험), 128.7879, 158.4848 (Model)
        t=24*3600,
        A_m=1.0,  # 초기 멤브레인 면적, 이후 업데이트됨
        target_concentration=target_concentration
    )
    
    @S401.add_specification(run=True)
    def update_ed_parameters():
        if hasattr(S401, 'fixed_A_m') and S401.fixed_A_m:
            print("🔹 민감도 분석 중 A_m 업데이트 건너뜀.")
            return
        """ED & AC Tank Update"""
        eff_ac = S401.outs[1]
        total_vfa_mass = eff_ac.imass['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid', 'LacticAcid'].sum()  # kg/hr
        total_vfa_mol = total_vfa_mass / 102.13  # kmol/hr (Valeric acid 60.05 g/mol)
        
        I = S401.j * S401.A_m  # 총 전류
        flux_dict = S401.calculate_flux(I)
        total_flux = sum(flux_dict.values())  # mol/(m2*s)
    
        # ✅ Q decided by ED flow rate
        Q = eff_ac.F_vol  # m³/hr
        
        # ✅ Updated Membrane Area (by Q)
        new_A_m = S401.calculate_membrane_area(total_vfa_mol, total_flux, Q)
        S401.A_m = new_A_m
        print(f"🔹 Updated ED Membrane Area: {S401.A_m:.4f} m²")

        # 🔹 Update tau of AC Tank
        update_ac_tau_based_on_target_concentration()
        # 🔹 Update tau of DC Tank (간단한 조정 함수 적용)
        update_dc_tau()

    # Method 3
    def update_ac_tau_based_on_target_concentration():
        """AC Tank의 배출량을 목표 농도에 맞춰 조정"""
        T302._design()  
        V_ac = T302.design_results['Volume']  # m³
        total_vfa_mass_ac = T302.outs[0].imass['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid', 'LacticAcid'].sum() # kg/hr
    
        # ✅ 목표 농도를 고려한 유출량 조정
        actual_concentration_ac = total_vfa_mass_ac / T302.outs[0].F_vol  # g/L
        print(f"📌 Actual Concentration in AC Tank: {actual_concentration_ac:.4f} g/L (Target: {target_concentration} g/L)")
        if actual_concentration_ac < target_concentration*4:
            print("🔹 Adjusting AC Tank discharge to maintain target concentration")
            T302.outs[0].F_mass *= 0.9  # 배출량 감소
        elif actual_concentration_ac > target_concentration*4:
            print("🔹 Adjusting AC Tank discharge to lower concentration")
            T302.outs[0].F_mass *= 1.1  # 배출량 증가
    
        new_tau = V_ac * target_concentration*4 / total_vfa_mass_ac
        T302.tau = new_tau
        print(f"✅ Updated AC Tank tau: {T302.tau:.4f} hr (adjusted for target concentration)")
    # Method 4
    # def update_ac_tau_based_on_target_concentration():
    #     """AC Tank의 배출 유량을 전체 유출량(물 포함) 기반으로 목표 농도에 맞춰 조정"""
    #     # AC 탱크 디자인 업데이트
    #     T302._design()
    #     V_ac = T302.design_results['Volume']  # AC 탱크 부피 (m³)
        
    #     # AC 탱크 outlet 스트림에서 VFA 질량 유량 (kg/hr)
    #     total_vfa_mass_ac = T302.outs[0].imass[['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid', 'LacticAcid']].sum()
        
    #     # AC 탱크 outlet 스트림의 전체 부피 유량 (m³/hr)
    #     total_flow = T302.outs[0].F_vol  
    #     # outlet 스트림의 밀도 (없다면 물의 밀도 1000 kg/m³로 가정)
    #     density = T302.outs[0].density if hasattr(T302.outs[0], 'density') else 1000  
    
    #     # 실제 VFA 농도 계산 (kg/m³); 1 kg/m³ = 1 g/L
    #     actual_concentration = total_vfa_mass_ac / (total_flow * density)
    #     print(f"Actual VFA concentration in AC Tank: {actual_concentration:.4f} kg/m³ (Target: {target_concentration/1000:.4f} kg/m³)")
        
    #     # 필요 시 배출량(F_mass) 조정 (여기서는 단순 예시로 10% 조정)
    #     if actual_concentration < target_concentration/1000:
    #         print("Adjusting AC Tank discharge: decreasing flow (increase τ)")
    #         T302.outs[0].F_mass *= 0.9
    #     elif actual_concentration > target_concentration/1000:
    #         print("Adjusting AC Tank discharge: increasing flow (decrease τ)")
    #         T302.outs[0].F_mass *= 1.1
        
    #     # 원하는 농도를 달성하기 위한 outlet 부피 유량 계산:
    #     # target concentration (kg/m³) = total_vfa_mass_ac / desired_flow  → desired_flow = total_vfa_mass_ac / target_concentration
    #     # (target_concentration을 kg/m³로 사용; 만약 target_concentration이 g/L이면 그대로 사용)
    #     desired_flow = total_vfa_mass_ac / (target_concentration/1000)  # m³/hr
        
    #     # AC 탱크의 체류시간은 탱크 부피(V_ac)를 desired_flow로 나눈 값
    #     new_tau = V_ac / desired_flow
    #     T302.tau = new_tau
    #     print(f"Updated AC Tank tau: {T302.tau:.4f} hr (based on total flow)")
    def update_dc_tau():
        """DC Tank의 체류시간을 AC Tank의 농도 변화에 따라 조정"""
        T301._design()  # DC Tank 디자인 업데이트
        
        V_dc = T301.design_results['Volume']  # DC Tank 부피 (m³)
        Q_dc = T301.outs[0].F_vol  # DC Tank 전체 배출 부피 유량 (m³/hr)
    
        # ✅ DC Tank 부피가 너무 작으면 최소값 설정
        if V_dc < 0.1:  
            print("⚠ Warning: DC Tank volume is too small. Assigning minimum volume (0.1 m³).")
            V_dc = 0.1  # 최소 부피 설정
        elif V_dc > 5:
            print("⚠ Warning: DC Tank volume is too large. Assigning maximum volume (5 m³).")
            V_dc = 5  # 최대 부피 제한
    
        # ✅ DC Tank 배출 스트림의 조성이 정의되지 않았을 경우 기본값 할당
        if T301.outs[0].isempty():
            print("⚠ Warning: DC Tank outlet stream is empty. Assigning default composition.")
            T301.outs[0].imass['Water'] = 1e-6  # 기본 성분 추가
            T301.outs[0].imass['AceticAcid'] = 1e-6  
    
        # ✅ DC Tank의 체류시간 업데이트 (총 부피 유량이 0이 아닐 경우)
        if Q_dc > 1e-6:
            # new_tau_dc = max(0.0001, min(10, V_dc / Q_dc))  # 최소 0.1hr, 최대 10hr로 제한
            new_tau_dc = V_dc / Q_dc
            T301.tau = new_tau_dc
            print(f"✅ Updated DC Tank tau: {T301.tau:.4f} hr (using F_vol)")
        else:
            print("⚠ Warning: DC Tank의 총 부피 유량이 0입니다. tau 업데이트를 건너뜁니다.")

    # --- 6. DC Output Handling (재순환 포함) ---
    S_DC = bst.Splitter(
        'S_DC',
        ins=S401-0,  # ED의 DC 출력
        outs=(recycle_dc, dc_output),
        split=0.9 # 90% 재순환, 10% 배출
    )
    
    # --- 8. AC Output Handling (재순환 포함) ---
    S_AC = bst.Splitter(
        'S_AC',
        ins=S401-1,  # ED의 AC 출력
        outs=(recycle_ac, 'ac_for_MEE'),
        split=0.9  # 90% 재순환, 10% MEE로 이동
    )

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
dc_tank = F.unit['T301']
ac_tank = F.unit['T302']
ed_unit = F.unit['S401']

# 결과 출력
print("--- DC/AC Tank and ED Design Information ---")
print(f"DC Tank Residence Time (tau): {dc_tank.tau} hr, Total Volume: {dc_tank.design_results['Total volume']:.4f} m³")
print(f"AC Tank Residence Time (tau): {ac_tank.tau} hr, Total Volume: {ac_tank.design_results['Total volume']:.4f} m³")
print(f"ED Required Membrane Area (A_m): {ed_unit.design_results['Membrane area']:.4f} m²")
print(f"ED Adjusted Current Density (j): {ed_unit.j:.4f} A/m²")
print(f"ED Power Consumption: {ed_unit.design_results['Power consumption']:.4f} W")