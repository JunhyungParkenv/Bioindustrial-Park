# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:30:14 2024

@author: Junhyung Park
"""

# %% Setup

import biosteam as bst
import thermosteam as tmo
from biosteam import units, Stream, SystemFactory
from biosteam.process_tools import SystemFactory
from biosteam import main_flowsheet
from biorefineries.cellulosic import units
from biorefineries.VFA import _chemicals
from biorefineries.VFA import _units
from biorefineries.VFA._chemicals import chems, chemical_groups, get_grouped_chemicals
from biorefineries.VFA._units import UASB, ED

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
tmo.settings.set_thermo(chems)

# # Add synonyms for easier referencing
# chems.set_synonym('H2O', 'Water')

# Define the system
@SystemFactory(
    ID='VFA_sys',
    ins=[dict(ID='feedstock', units='kg/hr')],  # Input stream
    outs=[
        dict(ID='separated_vfa', units='kg/hr'),  # Output VFA
        dict(ID='diluted_vfa', units='kg/hr')   # Spent stream
    ]
)
def create_VFA_sys(ins, outs):
    # Define input and output streams
    feedstock = ins[0]
    separated_vfa, spent_stream = outs

    # Initialize the input feedstock stream
    feedstock.imol['Water'] = 8751.4
    feedstock.imol['Glucose'] = 17632.44
    feedstock.price = 0.1

    # Define units
    # UASB fermentation
    R101 = UASB('R101', ins=feedstock, outs=('biogas', 'vfa_solution'))

    # ED separation
    S101 = ED('S101', ins=R101-1, outs=(separated_vfa, spent_stream))

    return [R101, S101]

#%%
# VFA 시스템 생성
VFA_sys = create_VFA_sys()
VFA_sys.diagram()
#%%
# ---------------------------
# TEA 객체 생성
# ---------------------------
template_tea = TEA(
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
    OSBL_units=[],
    labor_cost=1e6,  # 예시 비용
    labor_burden=0.9,
    property_insurance=0.007,
    maintenance=0.03,
)
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