#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Bioindustrial-Park: BioSTEAM's Premier Biorefinery Models and Results
# Copyright (C) 2022-2023, Sarang Bhagwat <sarangb2@illinois.edu> (this biorefinery)
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""

@author: sarangbhagwat

Created on Sun Aug 23 12:11:15 2020

This module is a modified implementation of modules from the following:
[1]	Bhagwat et al., Sustainable Production of Acrylic Acid via 3-Hydroxypropionic Acid from Lignocellulosic Biomass. ACS Sustainable Chem. Eng. 2021, 9 (49), 16659–16669. https://doi.org/10.1021/acssuschemeng.1c05441
[2]	Li et al., Sustainable Lactic Acid Production from Lignocellulosic Biomass. ACS Sustainable Chem. Eng. 2021, 9 (3), 1341–1351. https://doi.org/10.1021/acssuschemeng.0c08055
[3]	Cortes-Peña et al., BioSTEAM: A Fast and Flexible Platform for the Design, Simulation, and Techno-Economic Analysis of Biorefineries under Uncertainty. ACS Sustainable Chem. Eng. 2020, 8 (8), 3302–3310. https://doi.org/10.1021/acssuschemeng.9b07040

All units are explicitly defined here for transparency and easy reference.
Naming conventions:
    D = Distillation column
    AC = Adsorption column
    F = Flash tank or multiple-effect evaporator
    H = Heat exchange
    M = Mixer
    P = Pump (including conveying belt)
    R = Reactor
    S = Splitter (including solid/liquid separator)
    T = Tank or bin for storage
    U = Other units
Processes:
    100: Feedstock preprocessing
    200: Pretreatment
    300: Conversion
    400: Separation
    500: Wastewater treatment
    600: Storage
    700: Co-heat and power
    800: Cooling utility generation
    900: Miscellaneous facilities
    1000: Heat exchanger network

"""


# %% Setup


import biosteam as bst
import thermosteam as tmo
import flexsolve as flx
import numpy as np
from math import exp as math_exp
# from biosteam import main_flowsheet as F
# from copy import deepcopy
# from biosteam import System
from thermosteam import Stream
# from biorefineries.cornstover import CellulosicEthanolTEA
from biorefineries.TAL import units, facilities

from biorefineries.TAL._process_specification import ProcessSpecification
from biorefineries.TAL._general_process_specification import GeneralProcessSpecification

from biorefineries.TAL.process_settings import price, CFs
from biorefineries.TAL.utils import find_split, splits_df, baseline_feedflow
from biorefineries.TAL.chemicals_data import TAL_chemicals, chemical_groups, \
                                soluble_organics, combustibles
# from biorefineries.TAL.tea import TALTEA
from biorefineries.cornstover import CellulosicEthanolTEA as TALTEA
from biosteam import System, SystemFactory
from warnings import filterwarnings

from biorefineries.cellulosic import create_facilities
from biorefineries.sugarcane import create_juicing_system_up_to_clarification
from biorefineries.TAL.process_areas import create_TAL_fermentation_process,\
                        create_TAL_separation_adsorption_process,\
                        create_TAL_to_sorbic_acid_upgrading_process

from biorefineries.succinic.lca import LCA as TALLCA

filterwarnings('ignore')
IQ_interpolation = flx.IQ_interpolation
HeatExchangerNetwork = bst.HeatExchangerNetwork
# from hxn._heat_exchanger_network import HeatExchangerNetwork
Rxn = tmo.reaction.Reaction
ParallelRxn = tmo.reaction.ParallelReaction

# from lactic.hx_network import HX_Network

# # Do this to be able to show more streams in a diagram
flowsheet = bst.Flowsheet('TAL')
bst.main_flowsheet.set_flowsheet(flowsheet)

# Speeds up ShortcutDistillation
bst.units.ShortcutColumn.minimum_guess_distillate_recovery = 0

# Baseline cost year is 2016
bst.CE = 541.7

# Set default thermo object for the system
tmo.settings.set_thermo(TAL_chemicals)



# %% 
@SystemFactory(ID = 'TAL_sys')
def create_TAL_sys(ins, outs):

    # %% Feedstock
    
    # Sugarcane juicing subprocess
    sugarcane_juicing_sys = create_juicing_system_up_to_clarification()
    u = sugarcane_juicing_sys.flowsheet.unit
    s = sugarcane_juicing_sys.flowsheet.stream
    feedstock = s.sugarcane
    feedstock.F_mass = 446788.6418279631 # at baseline, produces KSA at ~26794 metric tonne / y
                                         # or ~20,000 metric tonne KSA eq. / year 

    #%% Fermentation
    fermentation_sys = create_TAL_fermentation_process(ins=(u.C201-0, 'CSL_fermentation', 'Acetate_fermentation'),
                                                   )
    
    # %% Separation
    Ethanol_desorption = Stream('Ethanol_desorption', units='kg/hr')
    separation_sys = create_TAL_separation_adsorption_process(ins=(fermentation_sys-1,
                                                                   Ethanol_desorption))

    
    #%% Upgrading
    ethanol_minimal = tmo.Stream('ethanol_minimal',)
    H2_hydrogenation = tmo.Stream('H2_hydrogenation',)
    KOH_hydrolysis = tmo.Stream('KOH_hydrolysis',)
    acetone_purification = tmo.Stream('acetone_purification',)
    fresh_catalyst_R401 = tmo.Stream('fresh_catalyst_R401', price=price['Ni-SiO2'])
    fresh_catalyst_R402 = tmo.Stream('fresh_catalyst_R402', price=price['Amberlyst-70'])
    upgrading_sys = create_TAL_to_sorbic_acid_upgrading_process(ins=(separation_sys-0,
                                                                     ethanol_minimal,
                                                                     H2_hydrogenation,
                                                                     KOH_hydrolysis,
                                                                     acetone_purification,
                                                                     fresh_catalyst_R401,
                                                                     fresh_catalyst_R402,
                                                                     ),
                                                                )
    
    # %% 
    
    # =============================================================================
    # Wastewater treatment streams
    # =============================================================================
    

    # =============================================================================
    # Wastewater treatment units
    # =============================================================================
    
    # Mix waste liquids for treatment
    M501 = bst.units.Mixer('M501', ins=(
                                        u.F301-1,
                                        separation_sys-2,
                                        upgrading_sys-1,
                                        '', '', '', '',
                                        ))
    
    wastewater_treatment_sys = bst.create_wastewater_treatment_system(
        ins=M501-0,
        mockup=True,
        area=500,
    )
    
    # Mix solid wastes to boiler turbogenerator
    M510 = bst.units.Mixer('M510', ins=(separation_sys-1, 
                                        u.U202-0,
                                        u.C202-0,
                                        ),
                            outs='wastes_to_boiler_turbogenerator')
    
    MX = bst.Mixer(400, ['', ''])
    
    # %% 
    
    # =============================================================================
    # Facilities streams
    # =============================================================================
    

    CSL_fresh = Stream('CSL_fresh', price=price['CSL'])
    Acetate_fresh = Stream('Acetate_fresh', price=price['Acetic acid'])
    
    ethanol_fresh_desorption = Stream('ethanol_fresh_desorption',  price=price['Ethanol'])
    ethanol_fresh_upgrading = Stream('ethanol_fresh_upgrading',  price=price['Ethanol'])
    
    hydrogen_fresh = Stream('hydrogen_fresh', price=price['Hydrogen'], P=101325*300)
    KOH_fresh = Stream('KOH_fresh', price=price['KOH'])
    acetone_fresh = Stream('acetone_fresh', price=price['Acetone'])

    system_makeup_water = Stream('system_makeup_water', price=price['Makeup water'])
    
    
    imbibition_water = Stream('imbibition_water', price=price['Makeup water'])
    rvf_wash_water = Stream('rvf_wash_water', price=price['Makeup water'])
    dilution_water = Stream('dilution_water', price=price['Makeup water'])

    # Potassium sorbate product
    KSA_product = Stream('KSA_product', units='kg/hr', price=7.)

    # =============================================================================
    # Facilities units
    # =============================================================================
    
    
    T601 = units.CSLstorageTank('T601', ins=CSL_fresh, outs=1-fermentation_sys)
    T601.line = 'CSL storage tank'
   
    T602 = bst.StorageTank('T602', ins=hydrogen_fresh, outs='h2_liquid')
    T602.line = 'H2 storage tank'
    T602_P = bst.IsenthalpicValve('T602_P', ins=T602-0, outs=(2-upgrading_sys,), P=101325, vle=False)
    @T602_P.add_specification()
    def T602_P_spec():
        T602_P.ins[0].mol[:] = T602_P.outs[0].mol[:]
        T602_P.run()
    
    T603 = bst.StorageTank('T603', ins=KOH_fresh,)
    T603.line = 'KOH storage tank'
    T603_P = bst.ConveyingBelt('T603_P', ins=T603-0, outs=3-upgrading_sys)
    
    T604 = bst.StorageTank('T604', ins=acetone_fresh,)
    T604.line = 'Acetone storage tank'
    T604_P = bst.Pump('T604_P', ins=T604-0, outs=4-upgrading_sys)
    
    T605 = bst.StorageTank('T605', ins=ethanol_fresh_upgrading,)
    T605.line = 'Ethanol storage tank 1'
    T605_P = bst.Pump('T605_P', ins=T605-0, outs=1-upgrading_sys)
    
    T606 = bst.StorageTank('T606', ins=ethanol_fresh_desorption,)
    T606.line = 'Ethanol storage tank 2'
    T606_P = bst.Pump('T606_P', ins=T606-0, outs=1-separation_sys)
    
    T607 = bst.StorageTank('T607', ins=Acetate_fresh,)
    T607.line = 'Sodium acetate storage tank'
    T607_P = bst.ConveyingBelt('T607_P', ins=T607-0, outs=2-fermentation_sys)
    
    
    # 7-day storage time, similar to ethanol's in Humbird et al.
    T620 = units.TALStorageTank('T620', ins=upgrading_sys-0, tau=7*24, V_wf=0.9,
                                          vessel_type='Floating roof',
                                          vessel_material='Stainless steel')
    
    
    
    T620.line = 'TALStorageTank'
    
    T620_P = units.TALPump('T620_P', ins=T620-0, outs=KSA_product)
    
    s = flowsheet.stream
    create_facilities(
        solids_to_boiler=M510-0,
        gas_to_boiler=wastewater_treatment_sys-0,
        process_water_streams=[
         imbibition_water,
         rvf_wash_water,
         dilution_water,
         system_makeup_water,
         # s.fire_water,
         # s.boiler_makeup_water,
         # s.CIP,
         # s.recirculated_chilled_water,
         # s.s.3,
         # s.cooling_tower_makeup_water,
         # s.cooling_tower_chemicals,
         ],
        feedstock=s.sugarcane,
        RO_water=wastewater_treatment_sys-2,
        recycle_process_water=MX-0,
        BT_area=700,
        area=900,
    )
    
    HXN = HeatExchangerNetwork('HXN1001',
                                                ignored=[
                                                        u.H401, 
                                                        u.H402, 
                                                        u.H403, 
                                                        u.R401,
                                                        u.R402,
                                                        u.R403,
                                                        u.T602_P,
                                                        # H404,
                                                        # F403,
                                                        # AC401.heat_exchanger_drying,
                                                        # AC401.heat_exchanger_regeneration,
                                                        # # F401.components['condenser'],
                                                        ],
                                              cache_network=False,
                                              )
    
    def HXN_no_run_cost():
        HXN.heat_utilities = []
        HXN._installed_cost = 0.
    
    # # To simulate without HXN, simply uncomment the following 3 lines:
    HXN._cost = HXN_no_run_cost
    HXN.energy_balance_percent_error = 0.
    HXN.new_HXs = HXN.new_HX_utils = []

# %% 

# =============================================================================
# Complete system
# =============================================================================

TAL_sys = create_TAL_sys()

def set_system_and_all_subsystems_tolerances(system_,
                               molar_tolerance=1e-1, 
                               relative_molar_tolerance=1e-3):
    system_.molar_tolerance = molar_tolerance
    system_.relative_molar_tolerance = relative_molar_tolerance
    for i in system_.path:
        if type(i)==System:
            set_system_and_all_subsystems_tolerances(i,
                                           molar_tolerance=molar_tolerance, 
                                           relative_molar_tolerance=relative_molar_tolerance)

set_system_and_all_subsystems_tolerances(TAL_sys)
# TAL_sys.set_tolerance(mol=1e-1, rmol=1e-3)

f = bst.main_flowsheet
u = f.unit
s = f.stream

feedstock = s.sugarcane
KSA_product = s.KSA_product
get_flow_tpd = lambda: (feedstock.F_mass-feedstock.imass['H2O'])*24/907.185



TEA_feeds = set([i for i in TAL_sys.feeds if i.price]+ \
    [i for i in TAL_sys.feeds if i.price])

TEA_products = set([i for i in TAL_sys.products if i.price]+ \
    [i for i in TAL_sys.products if i.price]+[KSA_product])
    
    
for ui in u:
    globals().update({ui.ID: ui})
    
u.BT701.ID = 'BT701'
u.CT901.ID = 'CT801'
u.CWP901.ID = 'CWP802'
u.CIP901.ID = 'CIP901'
u.ADP901.ID = 'ADP902'
u.FWT901.ID = 'FWT903'
u.PWC901.ID = 'PWC904'


BT = flowsheet('BT')

BT.natural_gas_price = 0.2527
BT.ins[4].price = price['Lime']

globals().update(flowsheet.to_dict())
# %%
# =============================================================================
# TEA
# =============================================================================

# TAL_tea = CellulosicEthanolTEA(system=TAL_sys, IRR=0.10, duration=(2016, 2046),
#         depreciation='MACRS7', income_tax=0.21, operating_days=0.9*365,
#         lang_factor=None, construction_schedule=(0.08, 0.60, 0.32),
#         startup_months=3, startup_FOCfrac=1, startup_salesfrac=0.5,
#         startup_VOCfrac=0.75, WC_over_FCI=0.05,
#         finance_interest=0.08, finance_years=10, finance_fraction=0.4,
#         # biosteam Splitters and Mixers have no cost, 
#         # cost of all wastewater treatment units are included in WWT_cost,
#         # BT is not included in this TEA
#         OSBL_units=(u.U101, u.WWT_cost,
#                     u.T601, u.T602, u.T603, u.T606, u.T606_P,
#                     u.CWP, u.CT, u.PWC, u.CIP, u.ADP, u.FWT, u.BT),
#         warehouse=0.04, site_development=0.09, additional_piping=0.045,
#         proratable_costs=0.10, field_expenses=0.10, construction=0.20,
#         contingency=0.10, other_indirect_costs=0.10, 
#         labor_cost=3212962*get_flow_tpd()/2205,
#         labor_burden=0.90, property_insurance=0.007, maintenance=0.03,
#         steam_power_depreciation='MACRS20', boiler_turbogenerator=u.BT)

# TAL_no_BT_tea = TAL_tea
get_flow_dry_tpd = lambda: (feedstock.F_mass-feedstock.imass['H2O'])*24/907.185
TAL_tea = TALTEA(system=TAL_sys, IRR=0.10, duration=(2016, 2046),
        depreciation='MACRS7', income_tax=0.21, 
        # operating_days=0.9*365,
        operating_days = 200,
        lang_factor=None, construction_schedule=(0.08, 0.60, 0.32),
        startup_months=3, startup_FOCfrac=1, startup_salesfrac=0.5,
        startup_VOCfrac=0.75, WC_over_FCI=0.05,
        finance_interest=0.08, finance_years=10, finance_fraction=0.4,
        # biosteam Splitters and Mixers have no cost, 
        # cost of all wastewater treatment units are included in WWT_cost,
        # BT is not included in this TEA
        OSBL_units=(
                    u.U501,
                    # u.T601, u.T602, 
                    # u.T601, u.T602, u.T603, u.T604,
                    # u.T606, u.T606_P,
                    u.BT701, u.CT801, u.CWP802, u.CIP901, u.ADP902, u.FWT903, u.PWC904,
                    ),
        warehouse=0.04, site_development=0.09, additional_piping=0.045,
        proratable_costs=0.10, field_expenses=0.10, construction=0.20,
        contingency=0.10, other_indirect_costs=0.10, 
        labor_cost=3212962*get_flow_dry_tpd()/2205,
        labor_burden=0.90, property_insurance=0.007, maintenance=0.03,
        steam_power_depreciation='MACRS20', boiler_turbogenerator=u.BT701)

TAL_no_BT_tea = TAL_tea

# # Removed because there is not double counting anyways.
# # Removes feeds/products of BT_sys from TAL_sys to avoid double-counting
# for i in BT_sys.feeds:
#     TAL_sys.feeds.remove(i)
# for i in BT_sys.products:
#     TAL_sys.products.remove(i)

# Boiler turbogenerator potentially has different depreciation schedule
# BT_tea = bst.TEA.like(BT_sys, TAL_no_BT_tea)
# BT_tea.labor_cost = 0

# Changed to MACRS 20 to be consistent with Humbird
# BT_tea.depreciation = 'MACRS20'
# BT_tea.OSBL_units = (BT,)



#%% Define unit groups

area_names = [
    'feedstock',
    'conversion',
    'separation & upgrading',
    'wastewater',
    'storage',
    'boiler & turbogenerator',
    'cooling utility facilities',
    'other facilities',
    'heat exchanger network',
]


unit_groups = bst.UnitGroup.group_by_area(TAL_sys.units)


unit_groups.append(bst.UnitGroup('natural gas (for steam generation)'))
unit_groups.append(bst.UnitGroup('natural gas (for product drying)'))

unit_groups.append(bst.UnitGroup('fixed operating costs'))


for i, j in zip(unit_groups, area_names): i.name = j
for i in unit_groups: i.autofill_metrics(shorthand=False, 
                                         electricity_production=False, 
                                         electricity_consumption=True,
                                         material_cost=True)
for i in unit_groups:
    if i.name == 'storage' or i.name=='other facilities' or i.name == 'cooling utility facilities':
        i.metrics[-1].getter = lambda: 0. # Material cost
    if i.name == 'cooling utility facilities':
        i.metrics[1].getter = lambda: 0. # Cooling duty
    if i.name == 'boiler & turbogenerator':
        i.metrics[-1] = bst.evaluation.Metric('Material cost', 
                                            getter=lambda: TAL_tea.utility_cost/TAL_tea.operating_hours, 
                                            units='USD/hr',
                                            element=None) # Material cost
        # i.metrics[-2].getter = lambda: BT.power_utility.rate/1e3 # Electricity consumption [MW]
        
for HXN_group in unit_groups:
    if HXN_group.name == 'heat exchanger network':
        HXN_group.filter_savings = False
        assert isinstance(HXN_group.units[0], HeatExchangerNetwork)


unit_groups[-3].metrics[-1] = bst.evaluation.Metric('Material cost', 
                                                    getter=lambda: BT.natural_gas_price * BT.natural_gas.F_mass, 
                                                    units='USD/hr',
                                                    element=None)

F402, F403, F404 = u.F402, u.F403, u.F404
unit_groups[-2].metrics[-1] = bst.evaluation.Metric('Material cost', 
                                                    getter=lambda: sum([i.utility_cost-i.power_utility.cost\
                                                                        for i in [F402, F403, F404]]), 
                                                    units='USD/hr',
                                                    element=None)

unit_groups[-1].metrics[-1] = bst.evaluation.Metric('Material cost', 
                                                    getter=lambda: TAL_tea.FOC/TAL_tea.operating_hours, 
                                                    units='USD/hr',
                                                    element=None)

# for i in unit_groups:
#     i.metric(i.get_net_electricity_production,
#                 'Net electricity production',
#                 'kW')




unit_groups_dict = {}
for i in unit_groups:
    unit_groups_dict[i.name] = i

BT = u.BT701
CT = u.CT801
CWP = u.CWP802
HXN = u.HXN1001


# %% 
# =============================================================================
# Simulate system and get results
# =============================================================================


def get_KSA_MPSP():
    for i in range(3):
        TAL_sys.simulate()
    for i in range(3):
        KSA_product.price = TAL_tea.solve_price(KSA_product)
    return KSA_product.price*KSA_product.F_mass/KSA_product.imass['KSA']

theoretical_max_g_TAL_per_g_glucose = 2*TAL_chemicals.TAL.MW/(3*TAL_chemicals.Glucose.MW)
spec = ProcessSpecification(
    evaporator = u.F301,
    pump = None,
    mixer = u.M304,
    heat_exchanger = u.M304_H,
    seed_train_system = [],
    seed_train = u.R303,
    reactor= u.R302,
    reaction_name='fermentation_reaction',
    substrates=('Xylose', 'Glucose'),
    products=('TAL',),
    
    spec_1=0.0815/theoretical_max_g_TAL_per_g_glucose, # mean of 0.074 and 0.089 g/g (range from Cao et al. 2022)
    spec_2=25.5, # mean of 0.074 and 0.089 g/g (range from Cao et al. 2022)
    spec_3=0.215, # mean of 0.074 and 0.089 g/g (range from Cao et al. 2022)

    
    xylose_utilization_fraction = 0.80,
    feedstock = feedstock,
    dehydration_reactor = None,
    byproduct_streams = [],
    HXN = u.HXN1001,
    maximum_inhibitor_concentration = 1.,
    # pre_conversion_units = process_groups_dict['feedstock_group'].units + process_groups_dict['pretreatment_group'].units + [u.H301], # if the line below does not work (depends on BioSTEAM version)
    # pre_conversion_units = TAL_sys.split(u.M304.ins[0])[0],
    pre_conversion_units = [],
    # !!! set baseline fermentation performance here
    # (ranges from Cao et al. 2022)
    baseline_yield = 0.0815/theoretical_max_g_TAL_per_g_glucose, # mean of 0.074 and 0.089 g/g 
    baseline_titer = 25.5, # mean of 23 and 28 g/L
    baseline_productivity = 0.215, # mean of 0.19 and 0.24 g/L/h
    
    # baseline_yield = 0.30,
    # baseline_titer = 25.,
    # baseline_productivity = 0.19,
    
    feedstock_mass = feedstock.F_mass,
    pretreatment_reactor = None)


spec.load_spec_1 = spec.load_yield
# spec.load_spec_2 = spec.load_titer
spec.load_spec_3 = spec.load_productivity

def M304_titer_obj_fn(water_to_sugar_mol_ratio):
    # M304, R302 = u.M304, u.R302
    M304.water_to_sugar_mol_ratio = water_to_sugar_mol_ratio
    M304.specifications[0]()
    M304_H._run()
    S302.specifications[0]()
    R303._run()
    T301._run()
    # R302.simulate()
    R302.specifications[0]()
    return R302.effluent_titer - R302.titer_to_load

def F301_titer_obj_fn(V):
    F301.V = V
    F301._run()
    H301._run()
    M304.specifications[0]()
    M304_H._run()
    S302.specifications[0]()
    R303._run()
    T301._run()
    # R302.simulate()
    R302.specifications[0]()
    return R302.effluent_titer - R302.titer_to_load

def load_titer_with_glucose(titer_to_load):
    spec.spec_2 = titer_to_load
    R302.titer_to_load = titer_to_load
    F301_titer_obj_fn(1e-4)
    if M304_titer_obj_fn(1e-4)<0: # if the slightest dilution results in too low a conc
        IQ_interpolation(F301_titer_obj_fn, 1e-4, 0.8, ytol=1e-4)
    # elif F301_titer_obj_fn(1e-4)>0: # if the slightest evaporation results in too high a conc
    else:
        F301_titer_obj_fn(1e-4)
        IQ_interpolation(M304_titer_obj_fn, 1e-4, 10000., ytol=1e-4) # for titers lower than 5 g/L with high yields, use an upper bound of 20000
    # else:
    #     raise RuntimeError('Unhandled load_titer case.')
    spec.titer_inhibitor_specification.check_sugar_concentration()
spec.load_spec_2 = load_titer_with_glucose

# path = (F301, R302)
# @np.vectorize
# def calculate_titer(V):
#     F301.V = V
#     for i in path: i._run()
#     return spec._calculate_titer()

# @np.vectorize   
# def calculate_MPSP(V):
#     F301.V = V
#     TAL_sys.simulate()
#     MPSP = TAL.price = TAL_tea.solve_price(TAL, TAL_no_BT_tea)
#     return MPSP

# vapor_fractions = np.linspace(0.20, 0.80)
# titers = calculate_titer(vapor_fractions)
# MPSPs = calculate_MPSP(vapor_fractions)
# import matplotlib.pyplot as plt
# plt.plot(vapor_fractions, titers)
# plt.show()

# plt.plot(titers, MPSPs)
# plt.show()   


    
# %% Full analysis
def simulate_and_print():
    get_KSA_MPSP()
    print('\n---------- Simulation Results ----------')
    MPSP_KSA = get_KSA_MPSP()
    per_kg_KSA_to_per_kg_SA = TAL_chemicals.PotassiumSorbate.MW/TAL_chemicals.SorbicAcid.MW
    print(f'MPSP is ${MPSP_KSA:.3f}/kg PotassiumSorbate')
    print(f'.... or ${MPSP_KSA*per_kg_KSA_to_per_kg_SA:.3f}/kg SorbicAcid')
    GWP_KSA, FEC_KSA = TAL_lca.GWP, TAL_lca.FEC
    print(f'GWP-100a is {GWP_KSA:.3f} kg CO2-eq/kg PotassiumSorbate')
    print(f'........ or {GWP_KSA*per_kg_KSA_to_per_kg_SA:.3f} kg CO2-eq/kg SorbicAcid')
    print(f'FEC is {FEC_KSA:.3f} MJ/kg PotassiumSorbate')
    print(f'... or {FEC_KSA*per_kg_KSA_to_per_kg_SA:.3f} MJ/kg SorbicAcid')
    GWP_KSA_without_electricity_credit, FEC_KSA_without_electricity_credit =\
        GWP_KSA - TAL_lca.net_electricity_GWP, FEC_KSA - TAL_lca.net_electricity_FEC
    print(f'GWP-100a without electricity credit is {GWP_KSA_without_electricity_credit:.3f} kg CO2-eq/kg PotassiumSorbate')
    print(f'................................... or {GWP_KSA_without_electricity_credit*per_kg_KSA_to_per_kg_SA:.3f} kg CO2-eq/kg SorbicAcid')
    print(f'FEC without electricity credit is {FEC_KSA_without_electricity_credit:.3f} MJ/kg PotassiumSorbate')
    print(f'.............................. or {FEC_KSA_without_electricity_credit*per_kg_KSA_to_per_kg_SA:.3f} MJ/kg SorbicAcid')
    # print(f'FEC is {get_FEC():.2f} MJ/kg TAL or {get_FEC()/TAL_LHV:.2f} MJ/MJ TAL')
    # print(f'SPED is {get_SPED():.2f} MJ/kg TAL or {get_SPED()/TAL_LHV:.2f} MJ/MJ TAL')
    # print('--------------------\n')

# simulate_and_print()
# TAL_sys.simulate()
get_KSA_MPSP()

#%% LCA
TAL_lca = TALLCA(TAL_sys, CFs, s.sugarcane, KSA_product, ['PotassiumSorbate',], 
                           [], u.CT801, u.CWP802, u.BT701, True,
                           credit_feedstock_CO2_capture=True, add_EOL_GWP=True)
# u.AC401.cycle_time = 4.
# u.AC401.drying_time = 0.5 #!!! Drying time is updated to this value (overwritten the value passed during initialization)

#%% Load specifications
spec.load_specifications(spec.baseline_yield, spec.baseline_titer, spec.baseline_productivity)
simulate_and_print()

# %% Diagram

bst.LABEL_PATH_NUMBER_IN_DIAGRAMS = True
TAL_sys.diagram('cluster')

# %% 

# =============================================================================
# For Monte Carlo and analyses
# =============================================================================

TAL_sub_sys = {
#     'feedstock_sys': (U101,),
#     'pretreatment_sys': (T201, M201, M202, M203, 
#                          R201, R201_H, T202, T203,
#                          F201, F201_H,
#                          M204, T204, T204_P,
#                          M205, M205_P),
#     'conversion_sys': (H301, M301, M302, R301, R302, T301),
    # 'separation_sys': (S401, M401, M401_P,
    #                     S402, 
    #                     # F401, F401_H, X401,
    #                     D401, D401_H, D401_P, S403,
    #                     M402_P, S403,
    #                     D403, D403_H, D403_P,
    #                     M501,
    #                     T606, T606_P, T607, T607_P)
                        # F402, F402_H, F402_P,
                        # D405, D405_H1, D405_H2, D405_P,
                        # M401, M401_P)
#     'wastewater_sys': (M501, WWT_cost, R501,
#                        M502, R502, S501, S502, M503,
#                        M504, S503, S504, M510),
#     'HXN': (HXN,),
#     'BT': (BT,),
#     'CT': (CT,),
#     'other_facilities': (T601, S601,
#                          T602, T603,
#                          T604, T604_P,
#                          T605, T605_P,
#                          T606, T606_P,
#                          PWC, CIP, ADP, FWT)
    }

# for unit in sum(TAL_sub_sys.values(), ()):
#     if not unit in TAL_sys.units:
#         print(f'{unit.ID} not in TAL_sys.units')

# for unit in TAL_sys.units:
#     if not unit in sum(TAL_sub_sys.values(), ()):
#         print(f'{unit.ID} not in TAL_sub_sys')

#%% TEA breakdown

def TEA_breakdown(print_output=False):
    metric_breakdowns = {i.name: {} for i in unit_groups[0].metrics}
    for ug in unit_groups:
        for metric in ug.metrics:
            # storage_metric_val = None
            if not ug.name=='storage':
                if ug.name=='other facilities':
                    metric_breakdowns[metric.name]['storage and ' + ug.name] = metric() + unit_groups_dict['storage'].metrics[ug.metrics.index(metric)]()
                else:
                    metric_breakdowns[metric.name][ug.name] = metric()
                    
                    
            # if ug.name=='natural gas':
            #     if metric.name=='Mat. cost':
            #         metric_breakdowns[metric.name][ug.name] = BT.natural_gas.F_mass*BT.natural_gas_price
            
            
            # else:
            #     storage_metric_val = metric()
                
    # print and return metric_breakdowns
    if print_output:
        for i in unit_groups[0].metrics:
            print(f"\n\n----- {i.name} ({i.units}) -----")
            metric_breakdowns_i = metric_breakdowns[i.name]
            for j in metric_breakdowns_i.keys():
                print(f"{j}: {format(metric_breakdowns_i[j], '.3f')}")
    return metric_breakdowns

TEA_breakdown()

# market price range: $6.51 - $6.74 /kg
