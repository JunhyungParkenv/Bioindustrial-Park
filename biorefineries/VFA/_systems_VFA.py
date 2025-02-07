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
        dict(ID='evaporated_water', units='kg/hr'),
        dict(ID='biogas', units='kg/hr'),
        dict(ID='U302_cell_mass', units='kg/hr')
    ]
)
def create_VFA_sys(ins, outs):
    """VFA Recovery System: Anaerobic digestion and Electrodialysis-based separation"""
    feedstock = ins[0]
    stored_vfa, dc_output, evaporated_water, biogas, U302_cell_mass = outs

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
        outs=(U302_cell_mass, 'vfa_filtered')
    )

    # --- 3. Split into inf_dc and inf_ac ---
    S302 = bst.Splitter(
        'S302',
        ins=U302-1,  # vfa_filtered
        outs=('fresh_dc', 'fresh_ac'),
        split=0.8  # 80% inf_dc, 20% inf_ac
    )

    # **🔹 목표 농도 설정: MEE로 들어가는 AC 스트림의 목표 농도 = 4000 g/L**
    target_concentration = 4000  # g/L

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
        tau=None  # 나중에 업데이트됨
    )

    # --- 7. Electrodialysis (ED) ---
    S401 = _units.ED(
        'S401',
        ins=(T301-0, T302-0),
        outs=('treated_dc', 'treated_ac'),
        j=11.375,  # 전류 밀도
        t=24*3600,  # 작동 시간 (초)
        A_m=1.0,  # 초기값, 나중에 업데이트됨
    )

    # **🔹 목표 농도를 맞추도록 ED 및 Tank 업데이트**
    @S401.add_specification(run=True)
    def update_ed_parameters():
        """ED 및 Tank 설정을 target_concentration에 맞게 업데이트"""
        eff_ac = S401.outs[1]  # AC 스트림
        total_vfa_mass = eff_ac.imass['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid'].sum()  # VFA 총 질량 (g)
        total_water_mass = eff_ac.imass['Water']  # 물 질량 (g)

        # 현재 AC 스트림의 실제 농도 (g/L)
        current_concentration = total_vfa_mass / total_water_mass * 1000  # g/L 변환

        # 목표 농도까지의 차이를 계산하여 조정
        if current_concentration < target_concentration:
            concentration_ratio = target_concentration / current_concentration
            S401.A_m *= concentration_ratio  # 멤브레인 면적 증가
            print(f"🔹 Updated ED Membrane Area: {S401.A_m:.4f} m²")

        # AC Tank 체류 시간 업데이트
        ac_tank = T302
        ac_tank.tau = total_vfa_mass / (eff_ac.F_vol + 1e-6)
        ac_tank._design()
        print(f"✅ Updated AC Tank tau: {ac_tank.tau:.4f} hr")

    # --- 8. AC Output Handling (재순환 포함) ---
    S_AC = bst.Splitter(
        'S_AC',
        ins=S401-1,  # ED의 AC 출력
        outs=(recycle_ac, 'ac_for_MEE'),
        split=0.1  # 10% 재순환, 90% MEE로 이동
    )

    # --- 9. Multi-Effect Evaporator (MEE) ---
    E101 = bst.MultiEffectEvaporator(
        'E101',
        ins=S_AC-1,  # `ac_for_MEE`가 4000 g/L 농도를 만족해야 함
        outs=('vfa_evaporated', evaporated_water),
        V=0,
        P=(101325, 73581, 50892, 32777, 20000)
    )

    # --- 10. 확인: MEE 입력 농도 검증 ---
    @E101.add_specification(run=True)
    def verify_MEE_input():
        """MEE에 들어가는 `ac_for_MEE`의 농도가 target_concentration을 만족하는지 확인"""
        ac_for_MEE = S_AC.outs[1]
        vfa_mass = ac_for_MEE.imass['AceticAcid', 'PropionicAcid', 'ButyricAcid'].sum()
        water_mass = ac_for_MEE.imass['Water']
        actual_concentration = vfa_mass / water_mass * 1000  # g/L 변환

        print(f"🔹 MEE Input Concentration: {actual_concentration:.2f} g/L (Target: {target_concentration} g/L)")
        if abs(actual_concentration - target_concentration) > 50:
            print("⚠ Warning: MEE 입력 농도가 목표 농도와 차이가 큼. ED 설정을 다시 조정하세요.")



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

# # Electrodialysis (ED) 유닛의 멤브레인 면적(A_m) 및 전류 밀도(j) 출력
# ED_unit = F.unit.S401  # ED 유닛 불러오기
# A_m = ED_unit.design_results['Membrane area']
# j = ED_unit.j
# # 결과 출력
# print(f"✅ Required Membrane Area (Aₘ): {A_m:.2f} m²")
# print(f"✅ Current Density (j): {j:.2f} A/m²")
#%%
# ✅ `VFA_sys` 시뮬레이션 실행 후 `S401` 유닛 가져오기
# S401 = VFA_sys.flowsheet.unit.S401  # ED 유닛 가져오기
# S401._design()  # ✅ 설계 값 업데이트
# ✅ Membrane Area 한 번만 출력
# print(f"✅ Optimal Membrane Area: {S401.A_m:.3f} m²")
#%% 📌 **Membrane Area vs. Current Density 관계 분석**
# j_values = np.linspace(1, 15, 10)  # 전류 밀도 범위 설정 (1~15 mA/cm²)
# results = []

# for j in j_values:
#     S401.j = j  
#     S401._run()  # ✅ ED 프로세스 실행
#     S401._design()  # ✅ 설계 값 업데이트
#     results.append((j, S401.A_m, S401.design_results['Total current'], S401.design_results['Power consumption']))

# # ✅ 데이터프레임 생성
# df = pd.DataFrame(results, columns=["Current Density (mA/cm²)", "Membrane Area (m²)", "Total Current (A)", "Power Consumption (W)"])

# # ✅ 그래프 출력
# plt.figure(figsize=(8, 5))
# plt.plot(df["Current Density (mA/cm²)"], df["Membrane Area (m²)"], marker="o", linestyle="-", label="Membrane Area")
# plt.xlabel("Current Density (mA/cm²)")
# plt.ylabel("Membrane Area (m²)")
# plt.title("Membrane Area vs. Current Density in ED")
# plt.grid(True)
# plt.legend()
# plt.show()
