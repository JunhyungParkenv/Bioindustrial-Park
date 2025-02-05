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
# from biorefineries.cornstover import CellulosicEthanolTEA as TemplateTEA
# from biorefineries.VFA._process_settings import price
# # Create and compile chemicals
# chems = tmo.Chemicals([])

# # Add water and other chemicals
# H2O = tmo.Chemical('H2O')
# Glucose = tmo.Chemical('Glucose')
# LacticAcid = tmo.Chemical('LacticAcid')
# ButyricAcid = tmo.Chemical('ButyricAcid')
# PropionicAcid = tmo.Chemical('PropionicAcid')
# AceticAcid = tmo.Chemical('AceticAcid')
# ValericAcid = tmo.Chemical('ValericAcid', search_ID='PentanoicAcid')  # Valeric Acid alias
# CO2 = tmo.Chemical('CO2')

# # Append chemicals to the `chems` object
# chems.extend([H2O, Glucose, LacticAcid, ButyricAcid, PropionicAcid, AceticAcid, ValericAcid, CO2])
# chems.compile()

# # Add synonyms for easier referencing
# chems.set_synonym('H2O', 'Water')
# ✅ **Process settings 불러오기**
load_preferences_and_process_settings()  # Flow 단위를 'kg/hr'로 설정

# Thermodynamic properties
tmo.settings.set_thermo(chems)

# Flowsheet Initialization
F = bst.Flowsheet('VFA_Recovery')
bst.main_flowsheet.set_flowsheet(F)
# %% System Definition
# @SystemFactory(
#     ID='VFA_sys',
#     ins=[dict(ID='feedstock', units='kg/hr')],
#     outs=[dict(ID='stored_vfa', units='kg/hr'),
#           dict(ID='waste_stream', units='kg/hr')]
# )
@SystemFactory(
    ID='VFA_sys',
    ins=[dict(ID='feedstock', units='kg/hr')],
    outs=[
        dict(ID='stored_vfa', units='kg/hr'),     # 저장된 VFA
        dict(ID='dc_output', units='kg/hr'),  # 배출수(폐수)
        dict(ID='evaporated_water', units='kg/hr'),  # 증발된 물
        dict(ID='biogas', units='kg/hr'),        # 기체 배출물 (메탄, CO2 등)
        dict(ID='U302_cell_mass', units='kg/hr')    # 고체 폐기물 (세포 잔재물 등)
    ]
)
def create_VFA_sys(ins, outs):
    """
    VFA Recovery System: Anaerobic digestion and Electrodialysis-based separation
    """
    # Define Input and Output Streams
    feedstock = ins[0]
    stored_vfa, dc_output, evaporated_water, biogas, U302_cell_mass = outs

    # --- Feedstock Initialization ---
    feedstock.imass['Water'] = 154570.91
    feedstock.imass['Glucose'] = 3154.51
    feedstock.price = 0.1  # Price per kg

    # --- 1. Anaerobic Digestion (UASB Reactor) ---
    R101 = _units.UASB('R101', ins=feedstock, outs=('vfa_solution', biogas))
    
    # print("R101 outputs:")
    # print(f"VFA solution: {R101.outs[0].show()}")
    # print(f"Biogas: {R101.outs[1].show()}")
    
    # --- 2. Solid-Liquid Separation ---
    U302 = _units.CellMassFilter(
        'U302',
        ins=R101-0,  # vfa_solution
        outs=(U302_cell_mass, 'vfa_filtered'),
        moisture_content=None,
        split=0
    )
    
    # print("U302 outputs:")
    # print(f"Cell mass: {U302.outs[0].show()}")
    # print(f"VFA filtered: {U302.outs[1].show()}")
    
    # --- 2.1 Split into inf_dc and inf_ac ---
    S302 = bst.Splitter(
        'S302',
        ins=U302-1,  # vfa_filtered
        outs=('fresh_dc', 'fresh_ac'),
        split=0.8  # 80% inf_dc, 20% inf_ac
    )
    
    # --- 3. Recycle Streams ---
    recycle_dc = bst.Stream('recycle_dc')
    recycle_ac = bst.Stream('recycle_ac')

    # --- 4. Mix Tanks ---
    T301 = _units.MixTank(
        'dc_tank',
        ins=(S302-0, recycle_dc),
        outs='tank_to_dc',
        tau=24
    )
    T302 = _units.MixTank(
        'ac_tank',
        ins=(S302-1, recycle_ac),
        outs='tank_to_ac',
        tau=6
    )
    
    # --- 5. Electrodialysis (ED) Separation ---
    S401 = _units.ED(
        'S401',
        ins=(T301-0, T302-0),  # 두 MixTank의 혼합 출력
        outs=('treated_dc', 'treated_ac'),
        j=11.375,       # 전류 밀도
        t=24*3600,      # 작동 시간 (초)
    )
    
    # --- 6. DC Output Handling (재순환 포함) ---
    S_DC = bst.Splitter(
        'S_DC',
        ins=S401-0,  # ED의 DC 출력
        outs=(recycle_dc, dc_output),
        split=0.9  # 50% 재순환, 50% 배출
    )

    # --- 7. AC Output Handling (재순환 포함) ---
    S_AC = bst.Splitter(
        'S_AC',
        ins=S401-1,  # ED의 AC 출력
        outs=(recycle_ac, 'ac_for_MEE'),
        split=0.9  # 50% 재순환, 50% MEE로 이동
    )
    
    # --- 8. Multi-Effect Evaporator (MEE) ---
    E101 = bst.MultiEffectEvaporator(
        'E101',
        ins=S_AC-1,  # ac_for_MEE
        outs=('vfa_evaporated', evaporated_water),
        V=1e-4,
        P=(101325, 73581, 50892, 32777, 20000)
    )
    
    # --- 5. Crystallization ---
    S201 = bst.BatchCrystallizer(
        'S201',
        ins=E101-0,
        outs='solid_vfa',
        tau=48,  # Residence time
        N=6,  # Number of crystallizers
        T=320.15  # Temperature
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
        tau=7*24  # Storage time
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
# ✅ `VFA_sys` 시뮬레이션 실행 후 `S401` 유닛 가져오기
S401 = VFA_sys.flowsheet.unit.S401  # ED 유닛 가져오기
S401._design()  # ✅ 설계 값 업데이트
# ✅ Membrane Area 한 번만 출력
print(f"✅ Optimal Membrane Area: {S401.A_m:.3f} m²")
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
