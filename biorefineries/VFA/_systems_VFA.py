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

# Thermodynamic properties
tmo.settings.set_thermo(chems)

# Flowsheet Initialization
flowsheet = bst.Flowsheet('VFA_Recovery')
bst.main_flowsheet.set_flowsheet(flowsheet)

# %% Feedstock Stream Definition
# Feedstock stream 수정
# feedstock = Stream(
#     'feedstock',
#     Water=154570.91,  # 물의 질량 흐름 (kg/hr)
#     Glucose=3154.51,  # Glucose의 질량 흐름 (kg/hr)
#     # FermMicrobe=8.,Lignin = 5., 
#     # SolubleLignin = 10., GlucoseOligomer = 5.,
#     units='kg/hr',
#     price=0.1  # Example price
# )

# feedstock.imol['Water'] = 154570.91 / chems.Water.MW  # kg -> kmol
# feedstock.imol['Glucose'] = 3154.51 / chems.Glucose.MW  # kg -> kmol
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
    
    print("R101 outputs:")
    print(f"VFA solution: {R101.outs[0].show()}")
    print(f"Biogas: {R101.outs[1].show()}")
    
    # --- 2. Solid-Liquid Separation ---
    U302 = _units.CellMassFilter(
        'U302',
        ins=R101-0,  # vfa_solution
        outs=(U302_cell_mass, 'vfa_filtered'),
        moisture_content=None,
        split=0
    )
    
    print("U302 outputs:")
    print(f"Cell mass: {U302.outs[0].show()}")
    print(f"VFA filtered: {U302.outs[1].show()}")
    
    # --- 2.1 Split into inf_dc and inf_ac ---
    S302 = bst.Splitter(
        'S302',
        ins=U302-1,  # vfa_filtered
        outs=('inf_dc', 'inf_ac'),
        split=0.8  # 80% inf_dc, 20% inf_ac
    )
    
    print("S302 outputs:")
    print(f"inf_dc: {S302.outs[0].show()}")
    print(f"inf_ac: {S302.outs[1].show()}")

    # # --- 3. Electrodialysis Separation ---
    # S401 = _units.ED(
    #     'S401',
    #     ins=(S302-0, S302-1),  # inf_dc, inf_ac
    #     outs=('dc_output', 'ac_output'),  # Outputs for DC and AC tanks
    #     j=11.375,  # Current density
    #     t=24*3600,  # Time
    #     target_ratio=0.8
    # )

    # # --- 3.1 DC Tank and AC Tank ---
    # T301 = _units.DC_Tank(
    #     'T301',
    #     ins=S401-0,  # dc_output
    #     outs=waste_stream,  # DC Tank output to waste stream
    #     tau=24  # Residence time in hours
    # )

    # T302 = _units.AC_Tank(
    #     'T302',
    #     ins=S401-1,  # ac_output
    #     outs=('ac_output_to_mee'),  # AC Tank output to MEE
    #     tau=6  # Residence time in hours
    # )

    # # --- 4. Evaporation ---
    # E101 = bst.MultiEffectEvaporator(
    #     'E101',
    #     ins=T302-0,  # AC Tank output to MEE
    #     outs=('vfa_evaporated', evaporated_water),
    #     V=0.1,
    #     V_definition='First-effect',
    #     P=(101325, 73581, 50892, 32777)
    # )
    # --- 3. Electrodialysis Separation ---
    S401 = _units.ED(
        'S401',
        ins=(S302-0, S302-1),  # inf_dc, inf_ac
        outs=(dc_output, 'ac_output'),  # Outputs for DC and AC
        j=11.375,  # Current density
        t=24*3600,  # Time in seconds
        target_ratio=0.8
    )
    
    # --- 4. Evaporation ---
    E101 = bst.MultiEffectEvaporator(
        'E101',
        ins=S401-1,  # ac_output directly to MEE
        outs=('vfa_evaporated', evaporated_water),
        V=0.1,
        V_definition='First-effect',
        P=(101325, 73581, 50892, 32777)
    )
    
    # --- 5. Crystallization ---
    S201 = bst.BatchCrystallizer(
        'S201',
        ins=E101-0,
        outs='solid_vfa',
        tau=24,  # Residence time
        N=2,  # Number of crystallizers
        T=305.15  # Temperature
    )

    # --- 6. Storage ---
    T101 = bst.StorageTank(
        'T101',
        ins=S201-0,
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