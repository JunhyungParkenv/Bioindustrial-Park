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
from biorefineries.cornstover import CellulosicEthanolTEA as TemplateTEA
from biorefineries.VFA._process_settings import price
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

# Thermodynamic properties
tmo.settings.set_thermo(chems)
flowsheet = main_flowsheet
flowsheet.clear()
flowsheet.set_flowsheet(bst.Flowsheet('VFA_Recovery'))

# %% Feedstock Stream Definition
# Feedstock stream 수정
feedstock = Stream(
    'feedstock',
    Water=154570.91,  # 물의 질량 흐름 (kg/hr)
    Glucose=3154.51,  # Glucose의 질량 흐름 (kg/hr)
    # FermMicrobe=8.,Lignin = 5., 
    # SolubleLignin = 10., GlucoseOligomer = 5.,
    units='kg/hr',
    price=0.1  # Example price
)

feedstock.imol['Water'] = 154570.91 / chems.Water.MW  # kg -> kmol
feedstock.imol['Glucose'] = 3154.51 / chems.Glucose.MW  # kg -> kmol
# %% System Definition

@SystemFactory(
    ID='VFA_sys',
    ins=[dict(ID='feedstock', units='kg/hr')],
    outs=[dict(ID='stored_vfa', units='kg/hr'),
          dict(ID='waste_stream', units='kg/hr')]
)
def create_VFA_sys(ins, outs):
    """
    VFA Recovery System: Anaerobic digestion and Electrodialysis-based separation
    """
    # Define Input and Output Streams
    feedstock = ins[0]
    stored_vfa, waste_stream = outs

    # # Feedstock initialization
    # feedstock.imol['Water'] = 8751.4
    # feedstock.imol['Glucose'] = 17632.44
    # feedstock.price = 0.1

    # --- 1. Anaerobic Digestion (UASB Reactor) ---
    R101 = _units.UASB('R101', ins=feedstock, outs=('biogas', 'vfa_solution'))
    
    print("R101 outputs:")
    print(f"Biogas: {R101.outs[0].show()}")
    print(f"VFA solution: {R101.outs[1].show()}")
    
    # --- 2. Solid-Liquid Separation ---
    U302 = _units.CellMassFilter(
        'U302',
        ins=R101-1,
        outs=('U302_cell_mass', 'vfa_filtered'),
        moisture_content=0.99,
        split=0.01
    )
    
    print("U302 outputs:")
    print(f"Cell mass: {U302.outs[0].show()}")
    print(f"VFA filtered: {U302.outs[1].show()}")
    
    # --- 2.1 Split into inf_dc and inf_ac using Splitter ---
    S302 = bst.Splitter(
        'S302',
        ins=U302-1,  # vfa_filtered
        outs=('inf_dc', 'inf_ac'),
        split=0.8  # 80%는 inf_dc, 20%는 inf_ac로 분배
    )
    
    print("S302 outputs:")
    print(f"inf_dc: {S302.outs[0].show()}")
    print(f"inf_ac: {S302.outs[1].show()}")

    # --- 3. Electrodialysis Separation ---
    S401 = _units.ED(
        'S401',
        ins=(S302-1, S302-0),  # inf_dc, inf_ac
        outs=('vfa_concentrate', waste_stream),
        j=11.375,          # 전류 밀도 [A/m²]
        t=24*3600,         # 시간 [초]
        target_ratio=0.8,  # 목표 농도 비율
        dc_tau=24          # DC 탱크 체류 시간 [hr]
    )
    
    # 출력 스트림 확인
    print("S401 outputs:")
    print(f"VFA concentrate: {S401.outs[0].show()}")
    print(f"Waste stream: {S401.outs[1].show()}")
    
    # --- 4. Evaporation ---
    E101 = bst.MultiEffectEvaporator(
        'E101', 
        ins=S401-0,
        outs=('vfa_evaporated', 'evaporated_water'),
        V=0.1,
        V_definition='First-effect',
        P=(101325, 73581, 50892, 32777)
    )
    
    # --- 5. Crystallization (Single Output) ---
    S201 = bst.BatchCrystallizer(
        'S201',
        ins=E101-0,
        outs='solid_vfa'  # 하나의 출력만 사용
    )
    
    # --- 6. Storage ---
    T101 = bst.StorageTank(
        'T101',
        ins=S201-0,  # 'solid_vfa'가 입력으로 전달됨
        outs=stored_vfa,
        tau=7*24
    )
    
    # # --- ED 유닛 결과 확인 ---
    # print(f"A_m: {S401.A_m:.4f} m²")
    # print(f"DC Tank HRT: {S401.dc_storage.tau} hr")
    # print(f"AC Tank HRT: {S401.ac_storage.tau} hr")
    # print(f"DC Tank Volume: {S401.dc_storage.design_results.get('Total volume', 'N/A')} m³")
    # print(f"AC Tank Volume: {S401.ac_storage.design_results.get('Total volume', 'N/A')} m³")
    
    
    return [R101, U302, S401, E101, S201, T101]

#%%
# VFA System
VFA_sys = create_VFA_sys()
VFA_sys.diagram()
#%%
VFA_sys.simulate()
VFA_sys.show()
#%%
# ---------------------------
# TEA
# ---------------------------
template_tea = TemplateTEA(
    system=VFA_sys,
    IRR=0.10,
    duration=(2016, 2046),
    depreciation='MACRS7',
    income_tax=0.21,
    operating_days=0.9 * 365,
    construction_schedule=(0.08, 0.60, 0.32),
    startup_months=3,
    startup_FOCfrac=1,
    startup_salesfrac=0.5,
    startup_VOCfrac=0.75,
    WC_over_FCI=0.05,
    finance_interest=0.08,
    finance_years=10,
    finance_fraction=0.4,
    labor_cost=1e6,  # Example cost
    labor_burden=0.9,
    property_insurance=0.007,
    maintenance=0.03,
)
#%%
# ---------------------------
# Simulation and Results
# ---------------------------

def simulate_and_calculate():
    # Simulate the system
    VFA_sys.simulate()

    # Calculate CAPEX and OPEX
    CAPEX = template_tea.FCI  # Fixed Capital Investment
    OPEX = template_tea.AOC  # Annual Operating Cost excluding depreciation

    # Print results
    print("\n----- Economic Results -----")
    print(f"CAPEX (Fixed Capital Investment): ${CAPEX:,.2f}")
    print(f"OPEX (Annual Operating Cost): ${OPEX:,.2f}")
    print("----------------------------\n")

simulate_and_calculate()
#%%
# ---------------------------
# 시뮬레이션 실행 함수
# ---------------------------
def get_product_stream_MPSP():
    for i in range(3):
        VFA_sys.simulate()
    for i in range(3):
        VFA_sys.outs[0].price = template_tea.solve_price(VFA_sys.outs[0])
    return VFA_sys.outs[0].price

def simulate_and_print():
    MPSP = get_product_stream_MPSP()
    print('\n---------- Simulation Results ----------')
    print(f'MPSP is ${MPSP:.3f}/kg')
    print('----------------------------------------\n')

simulate_and_print()
#%%
# ---------------------------
# 다이어그램 출력
# ---------------------------
VFA_sys.diagram('cluster')