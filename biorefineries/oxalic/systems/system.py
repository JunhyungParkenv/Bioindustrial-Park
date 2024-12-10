#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Bioindustrial-Park: BioSTEAM's Premier Biorefinery Models and Results
# Copyright (C) 2023-2024, Sarang Bhagwat <sarangb2@illinois.edu> (this biorefinery)
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
import contourplots
from thermosteam import Stream
from biorefineries.TAL import units

from biorefineries.TAL.models._process_specification import ProcessSpecification
# from biorefineries.TAL.models._general_process_specification import GeneralProcessSpecification

from biorefineries.TAL.process_settings import price, CFs, chem_index
# from biorefineries.TAL.utils import find_split, splits_df, baseline_feedflow
from biorefineries.TAL.chemicals_data import TAL_chemicals, chemical_groups, \
                                soluble_organics, combustibles
# from biorefineries.TAL.tea import TALTEA
from biorefineries.TAL.tea import TALTEA
from biosteam import SystemFactory
from warnings import filterwarnings

from biorefineries.cellulosic import create_facilities
from biorefineries.sugarcane import create_juicing_system_up_to_clarification
from biorefineries.TAL.process_areas import create_TAL_fermentation_process,\
                        create_TAL_separation_solubility_exploit_process,\
                        create_TAL_to_sorbic_acid_upgrading_process

from biorefineries.TAL._general_utils import call_all_specifications_or_run,\
                                                get_more_unit_groups,\
                                                add_metrics_to_unit_groups,\
                                                set_production_capacity,\
                                                TEA_breakdown,\
                                                update_facility_IDs

from biorefineries.succinic.lca import LCA as TALLCA

filterwarnings('ignore')
IQ_interpolation = flx.IQ_interpolation
HeatExchangerNetwork = bst.HeatExchangerNetwork

Rxn = tmo.reaction.Reaction
ParallelRxn = tmo.reaction.ParallelReaction

# # Do this to be able to show more streams in a diagram
flowsheet = bst.Flowsheet('TAL')
bst.main_flowsheet.set_flowsheet(flowsheet)

# Speeds up ShortcutDistillation
bst.units.ShortcutColumn.minimum_guess_distillate_recovery = 0

# Baseline cost year is 2016
# bst.CE = 541.7

# Baseline cost year is 2019
bst.CE = 607.5

# Set default thermo object for the system
tmo.settings.set_thermo(TAL_chemicals)

# sc_temp = create_juicing_system_up_to_clarification()

# wwt_temp = bst.create_high_rate_wastewater_treatment_system(
#     ins=0-sc_temp, 
#     )

# %% 
@SystemFactory(ID = 'create_sugarcane_to_TAL_solubility_based_sys')
def create_sugarcane_to_TAL_solubility_based_sys(ins, outs):

    # %% Feedstock
    
    # Sugarcane juicing subprocess
    sugarcane_juicing_sys = create_juicing_system_up_to_clarification()
    
    u = sugarcane_juicing_sys.flowsheet.unit
    s = sugarcane_juicing_sys.flowsheet.stream
    
    u.U201.diagram()
    sugarcane_juicing_sys.diagram('cluster')
    # u.U201.ins.append(u.M201-0)
    
    # u.M201-0-1-u.U201
    
    # sugarcane_juicing_sys.simulate(update_configuration=True)
    
    U101 = bst.Unit('U101', ins='', outs='')
    @U101.add_specification(run=False)
    def U101_spec():
        U101.outs[0].copy_like(U101.ins[0])
    
    feedstock = s.sugarcane
    feedstock_sink = feedstock.sink
    U101-0-0-feedstock_sink
    feedstock-0-U101
    feedstock_sink.ins[0].price = 0.
    
    feedstock.F_mass = 554171.74 # initial value; updated by spec.set_production_capacity
    
    # Update all prices to 2019$ using chemical indices
    # sugarcane biorefinery base year is 2019
    for sugarcane_sys_stream in list(s):
        sugarcane_sys_stream.price *= chem_index[2019]/chem_index[2019]
        
    #%% Fermentation
    fermentation_sys = create_TAL_fermentation_process(ins=(u.C201-0),
                                                   )
    
    #%% Separation
    # acetylacetone_decarboxylation_equilibrium = Stream('acetylacetone_decarboxylation_equilibrium', 
    #                                                    phase='l', units='kg/hr')
    # base_for_pH_control_decarboxylation = Stream('base_for_pH_control_decarboxylation', 
    #                                                    units='kg/hr')
    
    separation_sys = create_TAL_separation_solubility_exploit_process(ins=(fermentation_sys-1,),
                                                                  )
    
    # no evaporator before recycle as heating would decarboxylate TAL
    S403 = bst.Splitter('S403', 
                        ins=separation_sys-2, 
                        outs=('S403_recycled_supernatant', 'S403_to_WWT'),
                        split=0., # optimal split=0.
                        )
    
    @S403.add_specification(run=False)
    def S403_spec():
        S403._run()
        for i in S403.outs: i.phase = 'l'
    S403-0-2-separation_sys
    
    H420 = bst.HXutility('H420', ins=separation_sys-4, outs='cooled_solid_TAL', 
                         T=273.15+25.,)
    
    #%% Upgrading
    # ethanol_minimal = tmo.Stream('ethanol_minimal',)
    # H2_hydrogenation = tmo.Stream('H2_hydrogenation',)
    # KOH_hydrolysis = tmo.Stream('KOH_hydrolysis',)
    # acetone_purification = tmo.Stream('acetone_purification',)
    
    # upgrading_sys = create_TAL_to_sorbic_acid_upgrading_process(ins=(separation_sys-3,
    #                                                                  ethanol_minimal,
    #                                                                  H2_hydrogenation,
    #                                                                  KOH_hydrolysis,
    #                                                                  acetone_purification,
    #                                                                  ),
    #                                                             )
    
    
    
    # %% 
    
    # =============================================================================
    # Facilities streams
    # =============================================================================
    

    CSL_fresh = Stream('CSL_fresh', price=price['CSL'])
    Acetate_fresh = Stream('Acetate_fresh', price=price['Acetic acid'])
    DAP_fresh = Stream('DAP_fresh', price=price['DAP'])
    
    acetylacetone_fresh = Stream('Acetylacetone_fresh', price=price['PD'])
    base_decarboxylation_fresh = Stream('base_decarboxylation_fresh', price=price['Sodium hydroxide'])
    
    hydrogen_fresh = Stream('hydrogen_fresh', price=price['Hydrogen'], P=101325*300)
    KOH_fresh = Stream('KOH_fresh', price=price['KOH'])
    acetone_fresh = Stream('acetone_fresh', price=price['Acetone'])
    ethanol_fresh = Stream('ethanol_fresh', price=price['Ethanol'])
    
    system_makeup_water = Stream('system_makeup_water', price=price['Makeup water'])
    
    
    imbibition_water = Stream('imbibition_water', price=price['Makeup water'])
    rvf_wash_water = Stream('rvf_wash_water', price=price['Makeup water'])
    dilution_water = Stream('dilution_water', price=price['Makeup water'])

    # TAL product
    TAL_product = Stream('TAL_product', units='kg/hr', price=7.)

    # =============================================================================
    # Facilities units
    # =============================================================================
    
    
    T601 = units.StorageTank('T601', ins=CSL_fresh)
    T601.line = 'CSL storage tank'
    T601_P = bst.Pump('T601_P', ins=T601-0, outs=1-fermentation_sys)
    
    T607 = bst.StorageTank('T607', ins=Acetate_fresh,)
    T607.line = 'Sodium acetate storage tank'
    T607_P = bst.ConveyingBelt('T607_P', ins=T607-0, outs=2-fermentation_sys)
    
    T608 = bst.StorageTank('T608', ins=acetylacetone_fresh,)
    T608.line = 'Acetylacetone storage tank'
    T608_P = bst.Pump('T608_P', ins=T608-0, outs=1-separation_sys)
    
    T609 = bst.StorageTank('T609', ins=DAP_fresh,)
    T609.line = 'DAP storage tank'
    T609_P = bst.Pump('T609_P', ins=T609-0, outs=3-fermentation_sys)
    
    T610 = bst.StorageTank('T610', ins=base_decarboxylation_fresh,)
    T610.line = 'SodiumHydroxide storage tank'
    T610_P = bst.Pump('T610_P', ins=T610-0, outs=3-separation_sys)
    
    
    # T602 = bst.StorageTank('T602', ins=hydrogen_fresh, outs=2-upgrading_sys)
    # T602.line = 'H2 storage tank'
    
    # T603 = bst.StorageTank('T603', ins=KOH_fresh,)
    # T603.line = 'KOH storage tank'
    # T603_P = bst.ConveyingBelt('T603_P', ins=T603-0, outs=3-upgrading_sys)
    
    # T604 = bst.StorageTank('T604', ins=acetone_fresh,)
    # T604.line = 'Acetone storage tank'
    # T604_P = bst.Pump('T604_P', ins=T604-0, outs=4-upgrading_sys)
    
    # T605 = bst.StorageTank('T605', ins=ethanol_fresh,)
    # T605.line = 'Ethanol storage tank'
    # T605_P = bst.Pump('T605_P', ins=T605-0, outs=1-upgrading_sys)
    
    
    # 7-day storage time, similar to ethanol's in Humbird et al.
    T620 = units.StorageTank('T620', ins=H420-0, tau=7*24, V_wf=0.9,
                                          vessel_type='Floating roof',
                                          vessel_material='Stainless steel')
    
    
    
    T620.line = 'TALStorageTank'
    
    T620_P = bst.Pump('T620_P', ins=T620-0, outs=TAL_product)
    
    # %% 
    
    # =============================================================================
    # Wastewater treatment streams
    # =============================================================================
    
    # For aerobic digestion, flow will be updated in AerobicDigestion
    air_lagoon = Stream('air_lagoon', phase='g', units='kg/hr')
    
    # To neutralize nitric acid formed by nitrification in aerobic digestion
    # flow will be updated in AerobicDigestion
    # The active chemical is modeled as NaOH, but the price is cheaper than that of NaOH
    aerobic_caustic = Stream('aerobic_caustic', units='kg/hr', T=20+273.15, P=2*101325,
                              price=price['Caustics'])
    
    # =============================================================================
    # Wastewater treatment units
    # =============================================================================
    
    # Mix waste liquids for treatment
    M501 = bst.units.Mixer('M501', ins=(
                                        # separation_sys-2,
                                        S403-1,
                                        u.F301-1,
                                        u.F401_P1-0,
                                        u.D401_P1-0,
                                        ))
    M501.citrate_acetate_dissolution_rxns = ParallelRxn([
        Rxn('SodiumAcetate + H2O -> AceticAcid + NaOH', 'SodiumAcetate',   1.-1e-5),
        Rxn('SodiumCitrate + H2O -> CitricAcid + 3NaOH ', 'SodiumCitrate',   1.-1e-5),
        ])
    
    @M501.add_specification(run=False)
    def M501_citrate_acetate_dissolution_spec():
        M501._run()
        M501.citrate_acetate_dissolution_rxns(M501.outs[0].mol[:])
        
    # wastewater_treatment_sys = bst.create_wastewater_treatment_system(
    #     kind='conventional',
    #     ins=M501-0,
    #     mockup=True,
    #     area=500,
    # )
    
    wastewater_treatment_sys = bst.create_high_rate_wastewater_treatment_system(
        ins=M501-0, 
        area=500, 
        mockup=False,
        # skip_AeF=True,
        )
    bst.settings.thermo.chemicals.set_synonym('BoilerChems', 'DAP')
    
    # U503 = u.U503
    # @U503.add_specification(run=True)
    # def U503_spec():
    #     # U503.outs[3].empty() # this stream accumulates gases including CO2, empty before simulating unit
    #     for i in U503.outs: i.empty()
        
    # Mix solid wastes to boiler turbogenerator
    M510 = bst.units.Mixer('M510', ins=(
                                        separation_sys-1,
                                        # upgrading_sys-1,
                                        u.U202-0,
                                        u.C202-0,
                                        ),
                            outs='wastes_to_boiler_turbogenerator')
    @M510.add_specification(run=True)
    def M510_spec():
        for i in M510.ins: i.phase='l'
        
    MX = bst.Mixer(900, ['', ''])
    
    M503 = u.M503
    @M503.add_specification(run=False)
    def M503_spec():
        for i in M503.ins: i.phase='l'
        M503._run()
        for j in M503.outs: j.phase='l'
        
    #%%
    s = flowsheet.stream
    create_facilities(
        solids_to_boiler=M510-0,
        gas_to_boiler=wastewater_treatment_sys-1,
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
    
    #%%
    CWP803 = bst.ChilledWaterPackage('CWP803', agent=bst.HeatUtility.cooling_agents[-2])
    
    BT = u.BT701
    BT.natural_gas_price = price['Natural gas']
    BT.ins[4].price = price['Lime']
    
    HXN = bst.HeatExchangerNetwork('HXN1001',
                                                ignored=[
                                                        u.H401, 
                                                        u.C401, 
                                                        u.F402,
                                                        u.H420,
                                                        # u.H402, 
                                                        # u.R401,
                                                        # u.R402,
                                                        # u.R403,
                                                        # u.M304_H,
                                                        ],
                                              cache_network=False,
                                              )
    
    def HXN_no_run_cost():
        HXN.heat_utilities = []
        HXN._installed_cost = 0.
    
    # # To simulate without HXN, simply uncomment the following 3 lines:
    # HXN._cost = HXN_no_run_cost
    # HXN.energy_balance_percent_error = 0.
    # HXN.new_HXs = HXN.new_HX_utils = []

# %% 

# =============================================================================
# Complete system
# =============================================================================

TAL_sys = create_sugarcane_to_TAL_solubility_based_sys()

TAL_sys.set_tolerance(mol=1e-3, rmol=1e-3, subsystems=True)

f = bst.main_flowsheet
u = f.unit
s = f.stream

feedstock = s.sugarcane
TAL_product = s.TAL_product
get_flow_tpd = lambda: (feedstock.F_mass-feedstock.imass['H2O'])*24/907.185

for ui in u:
    globals().update({ui.ID: ui})

update_facility_IDs(TAL_sys)

BT = u.BT701
CT = u.CT801
CWP = u.CWP802
CWP2 = u.CWP803
HXN = u.HXN1001


globals().update(flowsheet.to_dict())

# %%
# =============================================================================
# TEA
# =============================================================================

get_flow_dry_tpd = lambda: (feedstock.F_mass-feedstock.imass['H2O'])*24/907.185
TAL_tea = TALTEA(system=TAL_sys, IRR=0.125, duration=(2019, 2049),
        depreciation='MACRS7', income_tax=0.21, 
        operating_days = 180,
        lang_factor=None, construction_schedule=(0.08, 0.60, 0.32),
        startup_months=3, startup_FOCfrac=1, startup_salesfrac=0.5,
        startup_VOCfrac=0.75, WC_over_FCI=0.05,
        finance_interest=0.08, finance_years=10, finance_fraction=0.4,
        # biosteam Splitters and Mixers have no cost, 
        OSBL_units=(
                    u.U501,
                    # u.T601, u.T602, 
                    # u.T601, u.T602, u.T603, u.T604,
                    # u.T606, u.T606_P,
                    u.BT701, u.CT801, u.CWP802, u.CWP803, u.CIP901, u.ADP902, u.FWT903, u.PWC904,
                    ),
        warehouse=0.04, site_development=0.09, additional_piping=0.045,
        proratable_costs=0.10, field_expenses=0.10, construction=0.20,
        contingency=0.10, other_indirect_costs=0.10, 
        labor_cost=3212962*get_flow_dry_tpd()/2205,
        labor_burden=0.90, property_insurance=0.007, maintenance=0.03,
        steam_power_depreciation='MACRS20', boiler_turbogenerator=u.BT701)

TAL_no_BT_tea = TAL_tea

#%%
# =============================================================================
# LCA
# =============================================================================

TAL_lca = TALLCA(system=TAL_sys, 
                 CFs=CFs, 
                 feedstock=feedstock, 
                 main_product=TAL_product, 
                 main_product_chemical_IDs=['TAL',], 
                 by_products=['PD'], 
                 cooling_tower=u.CT801, 
                 chilled_water_processing_units=[u.CWP802, u.CWP803], 
                 boiler=u.BT701, has_turbogenerator=True,
                 credit_feedstock_CO2_capture=True, 
                 add_EOL_GWP=True,
                 )
# TAL_sys.LCA = TAL_lca

#%% Define unit groups and their metrics


feedstock_acquisition_group = bst.UnitGroup('feedstock acquisition', units=[u.U101])
feedstock_juicing_group = f.juicing_sys.to_unit_group('feedstock juicing')
fermentation_group = f.TAL_fermentation_process.to_unit_group('fermentation')
separation_group = f.TAL_separation_solubility_exploit_process.to_unit_group('separation')
separation_group.units.extend([u.S403, u.H420])
# upgrading_group = bst.UnitGroup('upgrading')


unit_groups = [
    feedstock_acquisition_group,
    feedstock_juicing_group,
    fermentation_group,
    separation_group,
    # upgrading_group,
    ]

unit_groups += get_more_unit_groups(system=TAL_sys,
                         groups_to_get=['wastewater',
                                        'storage & other facilities',
                                        'boiler & turbogenerator',
                                        'cooling utility facilities',
                                        'other facilities',
                                        'heat exchanger network',
                                        # 'natural gas (for steam generation)',
                                        'natural gas (for product drying)',
                                        # 'chilled brine',
                                        'fixed operating cost',
                                        'electricity consumption',
                                        'heating duty',
                                        'excess electricity',
                                        ]
                         )


add_metrics_to_unit_groups(unit_groups=unit_groups, system=TAL_sys, TEA=TAL_tea, LCA=TAL_lca)

unit_groups_dict = {}
for i in unit_groups:
    unit_groups_dict[i.name] = i

cooling_facilities_unit_group = unit_groups_dict['cooling utility facilities']

for i in cooling_facilities_unit_group.metrics:
    if i.name.lower() in ('electricity consumption', 'power consumption',):
        i.getter = lambda: sum([ui.power_utility.rate for ui in cooling_facilities_unit_group.units])/1e3

# %% 
# =============================================================================
# Simulate system and get results
# =============================================================================

try: TAL_sys.simulate()
except: pass

def get_TAL_MPSP():
    for i in range(3):
        TAL_sys.simulate()
    for i in range(3):
        TAL_product.price = TAL_tea.solve_price(TAL_product)
    return TAL_product.price*TAL_product.F_mass/TAL_product.imass['TAL']

theoretical_max_g_TAL_per_g_SA = TAL_chemicals.TAL.MW/TAL_chemicals.SorbicAcid.MW

theoretical_max_g_TAL_per_g_glucose = 2*TAL_chemicals.TAL.MW/(3*TAL_chemicals.Glucose.MW)

theoretical_max_g_TAL_per_g_acetic_acid = 0.22218*TAL_chemicals.TAL.MW/(TAL_chemicals.AceticAcid.MW)

g_sodium_acetate_to_g_acetic_acid = TAL_chemicals.AceticAcid.MW/TAL_chemicals.SodiumAcetate.MW

# fermentation_yield_lower_limit = 0.163/theoretical_max_g_TAL_per_g_glucose # lowest yield with acetate spike reported in Markham et al. 2018
fermentation_yield_baseline = 35.9/(180.*theoretical_max_g_TAL_per_g_glucose+13.7*g_sodium_acetate_to_g_acetic_acid*theoretical_max_g_TAL_per_g_acetic_acid) # from Markham et al. 2018; 35.9 g-TAL/L from 180 g-glucose/L and 13.7 g-sodium_acetate/L
# fermentation_yield_upper_limit = 0.203/theoretical_max_g_TAL_per_g_glucose

desired_annual_production = (23_802/2) * theoretical_max_g_TAL_per_g_SA # pure metric ton / y # satisfy 50% of 2019 US demand for sobic acid with 100% TAL->sorbic acid conversion

# desired_annual_production = (23_802) * theoretical_max_g_TAL_per_g_SA # pure metric ton / y # satisfy 100% of 2019 US demand for sobic acid with 100% TAL->sorbic acid conversion

spec = ProcessSpecification(
    evaporator = u.F301,
    pump = None,
    mixer = u.M304,
    heat_exchanger = u.M304_H,
    seed_train_system = [],
    seed_train = u.R303,
    reactor= u.R302,
    reaction_name='fermentation_reaction', # pure metric ton / y
    substrates=('Xylose', 'Glucose'),
    products=('TAL',),
    
    desired_annual_production = desired_annual_production, 
    
    spec_1=fermentation_yield_baseline, # from Markham et al. 2018; 35.9 g-TAL/L from 180 g-glucose/L and 13.7 g-sodium_acetate/L
    spec_2=35.9, # from Markham et al. 2018
    spec_3=0.12, # from Markham et al. 2018

    
    xylose_utilization_fraction = 0.80,
    feedstock = feedstock,
    dehydration_reactor = None,
    byproduct_streams = [],
    HXN = u.HXN1001,
    maximum_inhibitor_concentration = 1.,
    # pre_conversion_units = process_groups_dict['feedstock_group'].units + process_groups_dict['pretreatment_group'].units + [u.H301], # if the line below does not work (depends on BioSTEAM version)
    # pre_conversion_units = TAL_sys.split(u.M304.ins[0])[0],
    pre_conversion_units = [],
    
    # (ranges from Cao et al. 2022)
    # baseline_yield = 0.0815/theoretical_max_g_TAL_per_g_glucose, # mean of 0.074 and 0.089 g/g (range from Cao et al. 2022)
    # baseline_titer = 25.5, # mean of 23 and 28 g/L (range from Cao et al. 2022)
    # baseline_productivity = 0.215, # mean of 0.19 and 0.24 g/L/h (range from Cao et al. 2022)
    
    
    # !!! set baseline fermentation performance here
    baseline_yield = fermentation_yield_baseline, # from Markham et al. 2018; 35.9 g-TAL/L from 180 g-glucose/L and 13.7 g-sodium_acetate/L
    baseline_titer = 35.9, # from Markham et al. 2018
    baseline_productivity = 0.12, # from Markham et al. 2018
    
    
    tolerable_HXN_energy_balance_percent_error = 2.,
    
    feedstock_mass = feedstock.F_mass,
    pretreatment_reactor = None)


spec.load_spec_1 = spec.load_yield
# spec.load_spec_2 = spec.load_titer
spec.load_spec_3 = spec.load_productivity

def clear_units(units_to_clear):
    for i in units_to_clear:
        for j in list(i.ins)+list(i.outs):
            j.empty()
        i.simulate()
        
    
def M304_titer_obj_fn(water_to_sugar_mol_ratio):
    M304.water_to_sugar_mol_ratio = water_to_sugar_mol_ratio
    call_all_specifications_or_run([M304, M304_H, S302, R303, T301, R302, 
                                    V301, K301,
                                    ])
    return R302.effluent_titer - R302.titer_to_load

def F301_titer_obj_fn(V):
    F301.V = V
    call_all_specifications_or_run([F301, F301_P,
                                    H301, 
                                    M304, M304_H, S302, R303, T301, R302, 
                                    V301, K301,
                                    ])
    return R302.effluent_titer - R302.titer_to_load

def load_titer_with_glucose(titer_to_load):
    # clear_units([V301, K301])
    F301_lb, F301_ub = 0., 0.8
    M304_lb, M304_ub = 0., 20000.  # for low-titer high-yield combinations, if infeasible, use a higher upper bound
    
    R302.acetate_target_loading = R302.acetate_target_loading_default
    spec.spec_2 = titer_to_load
    R302.titer_to_load = titer_to_load
    F301_titer_obj_fn(F301_lb)
    
    if M304_titer_obj_fn(M304_lb) < 0.: # if there is too low a conc even with no dilution
        IQ_interpolation(F301_titer_obj_fn, F301_lb, F301_ub, ytol=1e-3)
    # elif F301_titer_obj_fn(1e-4)>0: # if the slightest evaporation results in too high a conc
    elif M304_titer_obj_fn(M304_ub) > 0.:
        R302.acetate_target_loading = spec.spec_2
        IQ_interpolation(M304_titer_obj_fn, 
                         M304_lb,
                         M304_ub, 
                         ytol=1e-3)
    else:
        F301_titer_obj_fn(F301_lb)
        IQ_interpolation(M304_titer_obj_fn, 
                         M304_lb, 
                         M304_ub, 
                         ytol=1e-3)

    spec.titer_inhibitor_specification.check_sugar_concentration()
    
spec.load_spec_2 = load_titer_with_glucose

# %% Full analysis
per_kg_KSA_to_per_kg_SA = TAL_chemicals.PotassiumSorbate.MW/TAL_chemicals.SorbicAcid.MW

production_capacity_is_fixed = True
def simulate_and_print():
    if production_capacity_is_fixed: spec_set_production_capacity(spec.desired_annual_production, method='analytical')
    # set_production_capacity(25000, 'analytical')
    print('\n---------- Simulation Results ----------')
    MPSP_KSA = get_TAL_MPSP()
    print(f'MPSP is ${MPSP_KSA:.3f}/kg TAL')
    # print(f'.... or ${MPSP_KSA*per_kg_KSA_to_per_kg_SA:.3f}/kg SorbicAcid')
    GWP_KSA, FEC_KSA = TAL_lca.GWP, TAL_lca.FEC
    print(f'GWP-100a is {GWP_KSA:.3f} kg CO2-eq/kg TAL')
    # print(f'........ or {GWP_KSA*per_kg_KSA_to_per_kg_SA:.3f} kg CO2-eq/kg SorbicAcid')
    print(f'FEC is {FEC_KSA:.3f} MJ/kg TAL')
    # print(f'... or {FEC_KSA*per_kg_KSA_to_per_kg_SA:.3f} MJ/kg SorbicAcid')
    GWP_KSA_without_electricity_credit, FEC_KSA_without_electricity_credit =\
        GWP_KSA - TAL_lca.net_electricity_GWP, FEC_KSA - TAL_lca.net_electricity_FEC
    print(f'GWP-100a without electricity credit is {GWP_KSA_without_electricity_credit:.3f} kg CO2-eq/kg TAL')
    # print(f'................................... or {GWP_KSA_without_electricity_credit*per_kg_KSA_to_per_kg_SA:.3f} kg CO2-eq/kg SorbicAcid')
    print(f'FEC without electricity credit is {FEC_KSA_without_electricity_credit:.3f} MJ/kg TAL')
    # print(f'.............................. or {FEC_KSA_without_electricity_credit*per_kg_KSA_to_per_kg_SA:.3f} MJ/kg SorbicAcid')
    # print(f'FEC is {get_FEC():.2f} MJ/kg TAL or {get_FEC()/TAL_LHV:.2f} MJ/MJ TAL')
    # print(f'SPED is {get_SPED():.2f} MJ/kg TAL or {get_SPED()/TAL_LHV:.2f} MJ/MJ TAL')
    # print('--------------------\n')

# simulate_and_print()
# TAL_sys.simulate()
get_TAL_MPSP()

#%% Misc.

def get_non_gaseous_waste_carbon_as_fraction_of_TAL_GWP100():
    return sum([i.get_atomic_flow('C') for i in TAL_sys.products if i.F_mol 
                and ('l' in i.phases or 's' in i.phases or i.phase=='l') 
                and (not i==TAL_product)])/TAL_product.imass['TAL']/TAL_lca.GWP

#%% Load specifications
spec.load_specifications(spec.baseline_yield, spec.baseline_titer, spec.baseline_productivity)


# If,  during TRY analysis, you'd like to set production capacity constant rather than feedstock capacity, uncomment the following line:

def spec_set_production_capacity(
                        desired_annual_production=spec.desired_annual_production, # pure metric ton /y
                        method='analytical', # 'IQ_interpolation' or 'analytical'
                        system=TAL_sys,
                        TEA=None,
                        spec=spec,
                        product_stream=TAL_product, 
                        product_chemical_IDs=['TAL',],
                        feedstock_stream=feedstock,
                        feedstock_F_mass_range=[5000, 2000_000], # wet-kg/h)
                        ):
    set_production_capacity(
                            desired_annual_production=desired_annual_production, # pure metric ton /y
                            method=method, # 'IQ_interpolation' or 'analytical'
                            system=system,
                            TEA=TEA,
                            spec=spec,
                            product_stream=product_stream, 
                            product_chemical_IDs=product_chemical_IDs,
                            feedstock_stream=feedstock_stream,
                            feedstock_F_mass_range=feedstock_F_mass_range, # wet-kg/h
                            )
    
spec.set_production_capacity = spec_set_production_capacity

spec_set_production_capacity(
                        desired_annual_production=spec.desired_annual_production, # pure metric ton /y
                        )

simulate_and_print()

# %% Diagram

bst.LABEL_PATH_NUMBER_IN_DIAGRAMS = True
TAL_sys.diagram('cluster')

#%% TEA breakdown


TEA_breakdown(unit_groups_dict=unit_groups_dict,
              print_output=True,
              )

#%% TEA breakdown figure

file_to_save = 'TAL_breakdown_plot'
###### change operating cost unit labels $/h to MM$/y
for i in unit_groups:
    for j in i.metrics:
        if j.name == 'Operating cost':
            j.units = r"$\mathrm{MM\$}$" + '\u00b7y\u207b\u00b9'
######

df_TEA_breakdown = bst.UnitGroup.df_from_groups(
    unit_groups, fraction=True,
    scale_fractions_to_positive_values=True,
)

# totals=[sum([ui.metrics[i]() for ui in unit_groups])
#         for i in range(len(unit_groups[0].metrics))]

totals=[]
metrics = unit_groups[0].metrics
for i in range(len(metrics)):
    curr_total = 0.
    for ui in unit_groups:
        curr_total += ui.metrics[i]()
    if metrics[i].name=='Operating cost':
        # change total operating cost from $/h to MM$/y
        curr_total *= TAL_tea.operating_hours/1e6
    totals.append(curr_total)


contourplots.stacked_bar_plot(dataframe=df_TEA_breakdown, 
                 y_ticks = [-50, -25, 0, 25, 50, 75, 100],
                 y_label=r"$\bfCost$" + " " + r"$\bfand$" + " " +  r"$\bfUtility$" + " " +  r"$\bfBreakdown$", 
                 y_units = "%", 
                 colors=['#7BBD84', 
                         '#E58835', 
                         '#F7C652', 
                         '#63C6CE', 
                         # '#b00000', 
                         '#94948C', 
                         '#734A8C', 
                         '#D1C0E1', 
                         '#648496', 
                         # '#B97A57', 
                         '#D1C0E1', 
                         # '#F8858A', 
                         '#F8858A', 
                         # '#63C6CE', 
                         '#94948C', 
                         # '#7BBD84', 
                         '#b6fcd5', 
                         '#E58835', 
                         # '#648496',
                         '#b6fcd5',
                         ],
                 hatch_patterns=('\\', '//', '|', 'x',),
                 filename=file_to_save+'TEA_breakdown_stacked_bar_plot',
                 n_minor_ticks=4,
                 fig_height=5.5*1.1777*0.94*1.0975,
                 fig_width=10,
                 show_totals=True,
                 totals=totals,
                 sig_figs_for_totals=3,
                 units_list=[i.units for i in unit_groups[0].metrics],
                 totals_label_text=r"$\bfsum:$",
                 )