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
from biorefineries.VFA._units import UASB, ED
from biorefineries.cornstover import CellulosicEthanolTEA as TemplateTEA
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

# %% System Definition

@SystemFactory(
    ID='VFA_sys',
    ins=[
        dict(ID='feedstock', units='kg/hr')
    ],
    outs=[
        dict(ID='separated_vfa', units='kg/hr'),
        dict(ID='spent_stream', units='kg/hr')
    ]
)
def create_VFA_sys(ins, outs):
    """
    VFA Recovery System: Anaerobic digestion and Electrodialysis-based separation
    """
    # Define Input and Output Streams
    feedstock = ins[0]
    separated_vfa, spent_stream = outs

    # Feedstock initialization
    feedstock.imol['Water'] = 8751.4
    feedstock.imol['Glucose'] = 17632.44
    feedstock.price = 0.1

    # --- 1. Anaerobic Digestion (UASB Reactor) ---
    R101 = UASB('R101', ins=feedstock, outs=('biogas', 'vfa_solution'))
    
    # --- 2. Solid-Liquid Separation ---
    U302 = _units.CellMassFilter('U302', 
                          ins=R101-1, 
                          outs=('U302_cell_mass', 'U302_to_WWT'),
                          moisture_content=0.35, 
                          split=0.99)
    
    # --- 3. Electrodialysis Separation ---
    S401 = ED('S401', ins=U302-1, outs=(separated_vfa, spent_stream))
    
    # --- 4. Evaporation ---
    E101 = bst.bstMultiEffectEvaporator('E101', 
                                 ins=S401-0,
                                 outs=('concentrated_vfa', 'evaporated_water'),
                                 V=0.1, 
                                 V_definition='First-effect',
                                 P=(101325, 73581, 50892, 32777))
    
    # --- 5. Crystallization ---
    S201 = bst.BatchCrystallizer('S201', 
                             ins=E101-0, 
                             outs=('solid_vfa', 'mother_liquor'))
    
    # --- 6. Storage ---
    T101 = bst.StorageTank('T101', ins=S201-0, outs='stored_vfa', tau=7*24)
    
    # --- Connections ---
    T101-0-1-S401  # Recycle stream to ED for optimization
    
    return [R101, U302, S401, E101, S201, T101]

# %% Diagram and Summary

if __name__ == '__main__':
    vfa_sys = create_VFA_sys()
    vfa_sys.simulate()
    vfa_sys.diagram('thorough')

    # System Summary
    vfa_sys.show()
    bst.report()

#%%
# VFA 시스템 생성
VFA_sys = create_VFA_sys()
VFA_sys.diagram()
#%%
# ---------------------------
# TEA 객체 생성
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