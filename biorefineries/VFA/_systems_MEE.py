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
bst.main_flowsheet.clear()       # ← 기존 flowsheet 완전 삭제
# Flowsheet Initialization
F = bst.Flowsheet('VFA_Recovery')
bst.main_flowsheet.set_flowsheet(F)
# %% System Definition
# 시스템 정의
@SystemFactory(
    ID='VFA_sys',
    ins=[dict(ID='feedstock',      units='kg/hr')],
    outs=[
        dict(ID='stored_vfa',     units='kg/hr'),
        dict(ID='biogas',         units='kg/hr'),
        dict(ID='U302_cell_mass', units='kg/hr'),
    ]
)
# 변경
def create_MEE_sys(ins, outs):
    feedstock, = ins
    stored_vfa, biogas, U302_cell_mass = outs

    # 1) Feedstock 세팅
    feedstock.imass['Water']   = 154570.91
    feedstock.imass['Glucose'] = 3154.51
    feedstock.price = 0.1  # $/kg

    # 2) Anaerobic Digestion (UASB)
    R101 = _units.UASB('R101', ins=feedstock, outs=('vfa_solution', biogas))

    # 3) Solid–Liquid Separation (Cell mass 제거)
    U302 = _units.CellMassFilter(
        'U302',
        ins=R101-0,                  # vfa_solution
        outs=(U302_cell_mass,        # cell mass
              'vfa_filtered'),       # filtered VFA solution
        moisture_content=None,
        split=0.0
    )

    # 4) MEE: 전체 VFA 용액 농축
    E401 = bst.units.MultiEffectEvaporator(
        'E401',
        ins=U302-1,          # vfa_filtered
        outs=('mee_concentrate', 'evaporator_steam'),
        P=(101325, 73581, 50892, 32777),  # 압력 단계 (예시)
        V=0.5                             # 농축 인자 (예시)
    )

    # 5) Crystallization (Batch)
    S201 = bst.BatchCrystallizer(
        'S201',
        ins=E401-0,           # MEE 농축액
        outs=('solid_vfa',),  # 결정화 고형물
        tau=6,                # Residence time [hr]
        N=4,                  # Number of crystallizers
        T=273.15 + 0.25       # Temperature [K]
    )

    # 6) Dryer (Drum)
    D301 = bst.DrumDryer(
        'D301',
        ins=S201-0,           # 결정화 고형물
        outs=('dried_vfa',),  # 건조 후 고형물
        moisture_content=0.05,        # 목표 수분 함량 5%
        split={'Water': 0.95},        # 물 95% 제거
        T=343.15                      # 온도 [K]
    )

    # 7) Storage Tank
    T101 = bst.StorageTank(
        'T101',
        ins=D301-0,          # 건조 VFA
        outs=stored_vfa,     # 시스템 최종 출력
        tau=7 * 24           # 7일 [hr]
    )

#%%
# MEE System
MEE_sys = create_MEE_sys()  
MEE_sys.diagram('cluster', number=True, format='png')
#%%
MEE_sys.simulate()
MEE_sys.show()
#%%
mee_unit = F.unit['E401']
print("--- MEE Unit Design Results ---")
print("단계 수:", mee_unit.number_of_effects)
print("디자인 리절트 키들:", mee_unit.design_results.keys())
if 'Steam duty' in mee_unit.design_results:
    print(f"Steam duty: {mee_unit.design_results['Steam duty']:.3f} kg/hr")
