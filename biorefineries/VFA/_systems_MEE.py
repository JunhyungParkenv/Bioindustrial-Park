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
bst.process_tools.default() 
# ─── MEEWithTarget 클래스 정의 ───────────────────────────────────────────────
from biosteam.units import MultiEffectEvaporator
from biosteam.units.vacuum_system import VacuumSystem
MultiEffectEvaporator.vacuum_system_preference = 'Liquid-ring pump'
class MEEWithTarget(MultiEffectEvaporator):
    # 더 이상 클래스 속성 패치는 필요 없습니다
    def __init__(self, ID='', ins=None, outs=(), *,
                 P=(101325, 73581, 50892, 32777),
                 V_definition='Overall',
                 target_concentration=None,
                 vfa_IDs=None,
                 **kwargs):
        super().__init__(ID=ID, ins=ins, outs=outs,
                         P=P, V=0.0, V_definition=V_definition, **kwargs)
        if target_concentration is None or vfa_IDs is None:
            raise ValueError("target_concentration와 vfa_IDs 모두 필요합니다.")
        self.target_concentration = target_concentration
        self.vfa_IDs = vfa_IDs

        # @self.add_specification(run=True)
        # def _set_V_from_target():
        #     feed = self.ins[0]
        #     total_vfa_mass = sum(feed.imass[id] for id in self.vfa_IDs)
        #     Q = feed.F_vol
        #     conc_in = total_vfa_mass * 1e3 / (Q * 1e3)
        #     self.V = max(0.0, min(1.0,
        #         1 - conc_in / self.target_concentration))
        #     self._reload_components = True
        
        @self.add_specification(run=True)
        def _set_V_from_target():
            """목표 농도에 맞춰 V를 설정하되, 0 이하가 되면 아주 작은 값(eps)으로 클램프."""
            feed = self.ins[0]
            total_vfa_mass = sum(feed.imass[id] for id in self.vfa_IDs)
            Q = feed.F_vol
            conc_in = total_vfa_mass * 1e3 / (Q * 1e3)
            raw = 1 - conc_in / self.target_concentration
            eps = 1e-6
            # raw가 너무 작거나 음수가 되면 eps로, 1보다 크면 1로 클램프
            self.V = min(1.0, max(eps, raw))
            self._reload_components = True
    # def _design(self):
    #     # ① 먼저 부모 디자인을 수행
    #     super()._design()
    #     # ② 그 결과로 나온 'Volume' 을 vacuum_system 에 전달, 전기 구동 펌프로 재생성
    #     vol = self.design_results.get('Volume')
    #     # P_suction 은 마지막 농축액 스트림의 압력
    #     P_suc = self.outs[0].P
    #     # 전기펌프(예: Liquid-ring pump)로 강제 설정
    #     self.vacuum_system = VacuumSystem(
    #         self, 
    #         'Liquid-ring pump',
    #         vessel_volume=vol,
    #         P_suction=P_suc
    #     )
    def _design(self):
        # ① 부모 설계 수행
        super()._design()
        # ② 면적이 너무 작거나 음수면 cost correlation이 깨지니 최소값으로 clamp
        A = self.design_results.get('Area', 0.0)
        A_min = 13.94 * 0.0929  # 13.94 ft² → m² 환산
        if A < A_min:
            self.design_results['Area'] = A_min
        # ③ 진공펌프 설계 (원래 로직)
        vol = self.design_results.get('Volume')
        P_suc = self.outs[0].P
        self.vacuum_system = VacuumSystem(
            self, 'Liquid-ring pump',
            vessel_volume=vol,
            P_suction=P_suc
        )
load_preferences_and_process_settings()  # Flow 단위를 'kg/hr'로 설정

# Thermodynamic properties
tmo.settings.set_thermo(chems)

# ✅ **🔹 Global Variable for Target Concentration**
# target_concentration = 2.694  # g/L # 0.898 (ED -> DC) * 3 -> 
target_concentration = 15  # g/L # 0.898 (ED -> DC) * 3 -> / 14.05 g/L (F.T302.outs[0]), 1.42 g/L (F.S401.ins[0])
vfa_IDs = ['AceticAcid','PropionicAcid','ButyricAcid','ValericAcid']
bst.main_flowsheet.clear()       # ← 기존 flowsheet 완전 삭제
# Flowsheet Initialization
F = bst.Flowsheet('VFA_Recovery')
bst.main_flowsheet.set_flowsheet(F)
# %% System Definition
# 시스템 정의
@SystemFactory(
    ID='MEE_sys',
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
    E401 = MEEWithTarget(
        'E401',
        ins=U302-1,
        outs=('mee_concentrate', 'evaporator_steam'),
        P=(101325, 73581, 50892, 32777),
        V_definition='First-effect',
        target_concentration=target_concentration,
        vfa_IDs=vfa_IDs
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
# P 튜플 길이로 단계 수를 구함
print("단계 수 (effects):", len(mee_unit.P))