#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:57:09 2024

@author: wenjun
"""

import numpy as np
from chaospy import distributions as shape
import biosteam as bst
from biosteam.evaluation import Model, Metric
from biosteam.evaluation.evaluation_tools.parameter import Setter
from biorefineries.SAF._chemicals import SAF_chemicals
from biorefineries.SAF.systems_coprocessing import SAF_sys, F
from biorefineries.SAF._tea import create_SAF_coprocessing_tea
from biorefineries.SAF._process_settings import price, GWP_CFs, load_preferences_and_process_settings
from warnings import warn
from warnings import filterwarnings; filterwarnings('ignore')

_gal_per_m3 = 1000/3.785412
_L_per_gal = 3.785412
_GGE_per_J = 1 / 116090 / 1055.06 # 1 BTU = 1055.06 J, gasoline LHV is 116 090 BTU per gal from https://afdc.energy.gov/fuels/properties
_kJpersec_to_kJhr = 3600

__all__ = ('create_model')

#%%

sys = SAF_sys(material_storage=False, product_storage=False, WWTC=False, BoilerTurbo=False, hydrogenation_distillation=False,h2_purchase=False,opportunity_cost=True)

# @sys.add_bounded_numerical_specification(x0=0, x1=0.3, xtol=1e-4, ytol=100)
# def adjust_bagasse_to_boiler(fraction_burned):
#     F.S102.split[:] = 1 - fraction_burned
#     sys.simulate()
#     excess = F.BT._excess_electricity_without_natural_gas
#     if fraction_burned == 0 and excess > 0:
#         return 0
#     elif fraction_burned == 0.3 and excess < 0:
#         return 0
#     else:
#         return excess

BT_sys = bst.System('BT_sys', path=(F.BT,))

preprocessing = bst.UnitGroup('Preprocessing_group', units = [i for i in sys.units if i.ID[1]=='1'])
                                          
pretreatment = bst.UnitGroup('Pretreatment_group', units = [i for i in sys.units if i.ID[1]=='2'])
                                          
fermentation = bst.UnitGroup('Fermentation_group', units = [i for i in sys.units if i.ID[1]=='3'])
                                          
upgrading = bst.UnitGroup('Upgrading_group', units = [i for i in sys.units if i.ID[1]=='4'])
                                          
wastewater_treatment = bst.UnitGroup('WWT_group', units = (F.WWT,))

heat_exchange_network = bst.UnitGroup('HXN_group', units = (F.HXN,))

boiler_turbogenerator = bst.UnitGroup('BT_group', units = (F.BT,)) 

cooling_tower = bst.UnitGroup('CT_group', units = (F.CT,)) 

facilities_no_hu = bst.UnitGroup('Facilities_no_hu_group', units = (F.CIP,)) 


process_groups = [preprocessing, pretreatment, fermentation, upgrading,
                  wastewater_treatment, 
                  heat_exchange_network, 
                  boiler_turbogenerator,
                  cooling_tower, facilities_no_hu]

process_groups_dict = {}
for i in range(len(process_groups)):
    group = process_groups[i]
    process_groups_dict[group.name] = group    
       
load_preferences_and_process_settings(T='K',
                                      flow_units='kg/hr',
                                      N=100,
                                      P_units='Pa',
                                      CE=798, # Average 2023 https://toweringskills.com/financial-analysis/cost-indices/
                                      indicator='GWP100',
                                      electricity_EI=GWP_CFs['electricity'],
                                      electricity_price=price['electricity'],
                                      )
sys.set_tolerance(rmol=1e-6, mol=1e-5, maxiter=400)

tea_SAF = create_SAF_coprocessing_tea(sys=sys,steam_distribution=0.0, water_supply_cooling_pumping=0.0, 
                                      water_distribution=0.0, electric_substation_and_distribution=0.0,
                                      gas_supply_and_distribution=0.009, comminication=0.0, safety_installation=0.013,
                                      building=0.07, yard_works=0.0, contingency_new=0.1, land=0.0, labor_cost=3763935,
                                      sanitary_waste_disposal=0.0)

sys.operating_hours = tea_SAF.operating_days * 24


def set_price_of_streams():
    for i in [F.preprocessing_sys,
              F.pretreatment_sys,
              F.fermentation_sys,
              F.upgrading_sys,
              F.WWT,
              F.BT,
              F.CT,
              F.CWP,
              F.PWC,]:
        for j in i.ins:
            if j.ID in price.keys():
                j.price = price[j.ID]
        for k in i.outs:
            if k.ID in price.keys():
                k.price = price[k.ID]
    F.hydrogen.price=0


def set_GWP_of_streams(indicator):
    F.caustic.set_CF(key='GWP100', value=GWP_CFs['caustic']) # caustic in WWT
    for i in [F.preprocessing_sys,
              F.pretreatment_sys,
              F.fermentation_sys,
              F.upgrading_sys,
              F.WWT,
              F.BT,
              F.CT,
              F.CWP,
              F.PWC]:
        for j in i.ins:
            if j.ID in GWP_CFs.keys():                
                j.characterization_factors[indicator]= GWP_CFs[j.ID]            
        for k in i.outs:
            if k.ID in GWP_CFs.keys():                
                k.characterization_factors[indicator] = GWP_CFs[k.ID]

set_prices = set_price_of_streams()
set_GWP = set_GWP_of_streams(indicator='GWP100')


# For simplification 
feedstock = F.feedstock
ethanol_to_upgrading = F.ethanol_to_upgrading
ethanol_to_sold = F.ethanol_to_sold
jet_fuel = F.jet_fuel
diesel = F.diesel
gasoline = F.gasoline


natural_gas = F.natural_gas
BT = F.BT
HXN = F.HXN

get_annual_factor = lambda: tea_SAF.operating_days * 24

##### Functions to calculate all the metrics #####

# 1. Product characteristics

get_coprocessing_ratio = lambda: F.R404.ins[0].F_vol * 24 / 5595 # in m3; 5595 is capacity of diesel hydrotreater in PR 

get_ethanol_yield = lambda: F.ethanol_to_upgrading.F_vol * _gal_per_m3 * get_annual_factor() / 1e6 # in MMGal (million gallon)
get_jet_yield = lambda:  jet_fuel.F_vol * _gal_per_m3 * get_annual_factor() / 1e6
get_diesel_yield = lambda:  diesel.F_vol * _gal_per_m3 * get_annual_factor() / 1e6
get_gasoline_yield = lambda:  gasoline.F_vol * _gal_per_m3 * get_annual_factor() / 1e6
get_total_yield = lambda:  get_jet_yield() + get_diesel_yield() + get_gasoline_yield()
get_jet_vol_ratio = lambda:  get_jet_yield() / get_total_yield() * 100
get_jet_to_eth_ratio = lambda:  get_jet_yield() / get_ethanol_yield() * 100

get_ethanol_energy = lambda:  F.ethanol_to_upgrading.LHV * _GGE_per_J * get_annual_factor() / 1000 # in MM GGE
get_jet_energy = lambda:  jet_fuel.LHV * _GGE_per_J * get_annual_factor() / 1000
get_diesel_energy = lambda:  diesel.LHV * _GGE_per_J * get_annual_factor() / 1000
get_gasoline_energy = lambda:  gasoline.LHV * _GGE_per_J * get_annual_factor() / 1000
get_total_energy = lambda:  get_jet_energy() + get_diesel_energy() + get_gasoline_energy()
get_total_energy_2 = lambda: get_jet_energy() + get_diesel_energy() + get_gasoline_energy() + get_excess_power() * 3600000 * _GGE_per_J / 1000 # including eletricity credit

get_jet_energy_ratio = lambda: get_jet_energy() / get_total_energy()
get_jet_energy_ratio_2 = lambda: get_jet_energy() / get_total_energy_2() 
# get_diesel_energy_ratio = lambda: get_diesel_energy() / get_total_energy()
# get_gasoline_energy_ratio = lambda: get_gasoline_energy() / get_total_energy()


# 2. TEA

get_installed_cost = lambda: tea_SAF.installed_equipment_cost / 1e6
get_installed_cost_OSBL = lambda: sum(i.installed_cost for i in tea_SAF.OSBL_units) / 1e6
get_installed_cost_ISBL = lambda: get_installed_cost() - get_installed_cost_OSBL()
get_DPI = lambda: tea_SAF.DPI / 1e6
get_TDC = lambda: tea_SAF.TDC / 1e6
get_FCI = lambda: tea_SAF.FCI / 1e6
get_TCI = lambda: tea_SAF.TCI / 1e6
get_FOC = lambda: tea_SAF.FOC / 1e6
get_VOC = lambda: tea_SAF.VOC / 1e6
get_AOC = lambda: tea_SAF.AOC / 1e6 # Excludes electricity credit

get_MPSP = lambda: tea_SAF.solve_price(F.jet_fuel)  # $/kg

_jet_price_conversion_index_vol = lambda: _L_per_gal * jet_fuel.rho / 1000 # from $/kg to $/gal
_jet_price_conversion_index_energy = lambda: jet_fuel.F_mass / jet_fuel.LHV * 1000000 # from $/kg to $/GJ

get_MPSP_per_gallon = lambda: get_MPSP() * _jet_price_conversion_index_vol()

get_jet_price_per_GJ = lambda: get_MPSP() * _jet_price_conversion_index_energy()

kerosene_price_per_GJ = 2.95/0.1899  # price is $2.95/gal (2022-202405 average) from https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EER_EPJK_PF4_RGC_DPG&f=M; LHV of jet-kerosene is 5.670 million Btu per barrel = 0.189910053 GJ/gallon from https://www.eia.gov/totalenergy/data/monthly/pdf/sec12.pdf
    
get_NPV = lambda: tea_SAF.NPV
solve_IRR = lambda: tea_SAF.solve_IRR()

get_material_cost = lambda: tea_SAF.material_cost / 1e6 #  Excludes electricity credit but includes the money spent on ash disposal 
get_annual_sale = lambda: tea_SAF.sales / 1e6

# Electricity 
# BT.power_utility.production - BT.power_utility.consumption = -BT.power_utility.rate
# i.power_utility.rate = i.power_utility.power; meaning requirement
get_excess_power = lambda: -sum(i.power_utility.rate for i in sys.units) * sys.operating_hours
electricity_price = price['electricity']
get_electricity_credit = lambda: get_excess_power() * electricity_price / 1e6

# In per gal of jet
get_cost_electricity_credit = lambda: get_excess_power() * electricity_price / jet_fuel_gal_per_year()



metrics = [Metric('Coprocessing ratio', get_coprocessing_ratio, ''),
           Metric('Installed cost', get_installed_cost, '10^6 $'),
           Metric('Installed cost OSBL', get_installed_cost_OSBL, '10^6 $'),
           Metric('Installed cost ISBL', get_installed_cost_ISBL, '10^6 $'),
           Metric('DPI', get_DPI, '10^6 $'),
           Metric('TDC', get_TDC, '10^6 $'),
           Metric('FCI', get_FCI, '10^6 $'),
           Metric('TCI', get_TCI, '10^6 $'),
           Metric('FOC', get_FOC, '10^6 $/yr'),
           Metric('VOC', get_VOC, '10^6 $/yr'),
           Metric('AOC', get_AOC, '10^6 $/yr'),
           
           Metric('Minimum selling price', get_MPSP_per_gallon, '$/gal'),
           Metric('Jet volume yield', get_jet_yield, '10^6 Gal/yr'),
           Metric('Total volume yield', get_total_yield, '10^6 Gal/yr'),
           Metric('Jet volume ratio', get_jet_vol_ratio, '%'),
           Metric('Jet energy ratio', get_jet_energy_ratio, '%'),
           Metric('Jet volume ratio to ethanol', get_jet_to_eth_ratio, '%'),
           
           Metric('Annual material cost', get_material_cost, '10^6 $/yr'),
           Metric('Annual product sale', get_annual_sale, '10^6 $/yr'),
           Metric('Annual electricity credit', get_electricity_credit, '10^6 $/yr'),
           Metric('Electricity credit to jet', get_cost_electricity_credit, '$/gal'),]





# # Breakdowns by process groups
# # index in per gallon jet fuel
jet_fuel_gal_per_hr = lambda: jet_fuel.F_vol * _gal_per_m3
jet_fuel_gal_per_year = lambda: jet_fuel_gal_per_hr() * sys.operating_hours

# def get_group_heating_demand(group):
#     return sum([sum([hu.duty for hu in unit.heat_utilities if hu.duty>0 and hu.flow>0.]) for unit in group.units])
# def get_group_cooling_demand(group):
#     return sum([sum([hu.duty for hu in unit.heat_utilities if hu.duty<0 and hu.flow>0.]) for unit in group.units])

# # Heating duty in (kJ/hr)/1000/(gal/hr)=MJ/gal
# metrics.extend((Metric('Total heating demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty>0 and hu.flow>0.]) for unit in \
#                                          sys.units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('Preprocessing_group-heating demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty>0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['Preprocessing_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('Pretreatment_group-heating demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty>0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['Pretreatment_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('Fermentation_group-heating demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty>0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['Fermentation_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('Upgrading_group-heating demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty>0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['Upgrading_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('WWT_group-heating demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty>0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['WWT_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('HXN_group-heating demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty<0 and hu.flow<0.]) for unit in \
#                                          process_groups_dict['HXN_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('BT_group-heating demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty>0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['BT_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('CT_group-heating demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty>0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['CT_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('Facilities_no_hu_group-heating demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty>0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['Facilities_no_hu_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# # metrics.extend((Metric('CCS-heating demand', 
# #                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
# #                                               if hu.duty>0 and hu.flow>0.]) for unit in \
# #                                          process_groups_dict['CCS'].units])/jet_fuel_gal_per_hr(),
# #                       'MJ/gal'),))


    
# # Cooling duty
# metrics.extend((Metric('Total cooling demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty<0 and hu.flow>0.]) for unit in \
#                                          sys.units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('Preprocessing_group-cooling demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty<0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['Preprocessing_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('Pretreatment_group-cooling demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty<0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['Pretreatment_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('Fermentation_group-cooling demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty<0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['Fermentation_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('Upgrading_group-cooling demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty<0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['Upgrading_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('WWT_group-cooling demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty<0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['WWT_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('HXN_group-cooling demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty>0 and hu.flow<0.]) for unit in \
#                                          process_groups_dict['HXN_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('BT_group-cooling demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty<0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['BT_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('CT_group-cooling demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty<0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['CT_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# metrics.extend((Metric('Facilities_no_hu_group-cooling demand', 
#                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
#                                               if hu.duty<0 and hu.flow>0.]) for unit in \
#                                          process_groups_dict['Facilities_no_hu_group'].units])/jet_fuel_gal_per_hr(),
#                       'MJ/gal'),))
# # metrics.extend((Metric('CCS-cooling demand', 
# #                       lambda: 0.001*sum([sum([hu.duty for hu in unit.heat_utilities \
# #                                               if hu.duty<0 and hu.flow>0.]) for unit in \
# #                                          process_groups_dict['CCS'].units])/jet_fuel_gal_per_hr(),
# #                       'MJ/gal'),))
    
    
# # Installed equipment cost
# metrics.extend((Metric('Preprocessing_group - installed equipment cost',
#                        lambda:process_groups_dict['Preprocessing_group'].get_installed_cost(),
#                        '10^6 $'),))
# metrics.extend((Metric('Pretreatment_group - installed equipment cost',
#                        lambda:process_groups_dict['Pretreatment_group'].get_installed_cost(),
#                        '10^6 $'),))
# metrics.extend((Metric('Fermentation_group - installed equipment cost',
#                        lambda:process_groups_dict['Fermentation_group'].get_installed_cost(),
#                        '10^6 $'),))
# metrics.extend((Metric('Upgrading_group - installed equipment cost',
#                        lambda:process_groups_dict['Upgrading_group'].get_installed_cost(),
#                        '10^6 $'),))
# metrics.extend((Metric('WWT_group - installed equipment cost',
#                        lambda:process_groups_dict['WWT_group'].get_installed_cost(),
#                        '10^6 $'),))
# metrics.extend((Metric('HXN_group - installed equipment cost',
#                        lambda:process_groups_dict['HXN_group'].get_installed_cost(),
#                        '10^6 $'),))
# metrics.extend((Metric('BT_group - installed equipment cost',
#                        lambda:process_groups_dict['BT_group'].get_installed_cost(),
#                        '10^6 $'),))
# metrics.extend((Metric('CT_group - installed equipment cost',
#                        lambda:process_groups_dict['CT_group'].get_installed_cost(),
#                        '10^6 $'),))
# metrics.extend((Metric('Facilities_no_hu_group - installed equipment cost',
#                        lambda:process_groups_dict['Facilities_no_hu_group'].get_installed_cost(),
#                        '10^6 $'),))
# # metrics.extend((Metric('CCS-installed equipment cost', 
# #                       lambda: process_groups_dict['CCS'].get_installed_cost(),
# #                        '10^6 $'),))


# # Power utility demand in MW/gal
# metrics.extend((Metric('Preprocessing_group - power utility demand',
#                        lambda:process_groups_dict['Preprocessing_group'].get_electricity_consumption()/jet_fuel_gal_per_hr(),
#                        'MW/gal'),))
# metrics.extend((Metric('Pretreatment_group - power utility demand',
#                        lambda:process_groups_dict['Pretreatment_group'].get_electricity_consumption()/jet_fuel_gal_per_hr(),
#                        'MW/gal'),))
# metrics.extend((Metric('Fermentation_group - power utility demand',
#                        lambda:process_groups_dict['Fermentation_group'].get_electricity_consumption()/jet_fuel_gal_per_hr(),
#                        'MW/gal'),))
# metrics.extend((Metric('Upgrading_group - power utility demand',
#                        lambda:process_groups_dict['Upgrading_group'].get_electricity_consumption()/jet_fuel_gal_per_hr(),
#                        'MW/gal'),))
# metrics.extend((Metric('WWT_group - power utility demand',
#                        lambda:process_groups_dict['WWT_group'].get_electricity_consumption()/jet_fuel_gal_per_hr(),
#                        'MW/gal'),))
# metrics.extend((Metric('HXN_group - power utility demand',
#                        lambda:process_groups_dict['HXN_group'].get_electricity_consumption()/jet_fuel_gal_per_hr(),
#                        'MW/gal'),))
# metrics.extend((Metric('BT_group - power utility demand',
#                        lambda:process_groups_dict['BT_group'].get_electricity_consumption()/jet_fuel_gal_per_hr(),
#                        'MW/gal'),))
# metrics.extend((Metric('CT_group - power utility demand',
#                        lambda:process_groups_dict['CT_group'].get_electricity_consumption()/jet_fuel_gal_per_hr(),
#                        'MW/gal'),))
# metrics.extend((Metric('Facilities_no_hu_group - power utility demand',
#                        lambda:process_groups_dict['Facilities_no_hu_group'].get_electricity_consumption()/jet_fuel_gal_per_hr(),
#                        'MW/gal'),))  
# # metrics.extend((Metric('CCS - power utility demand',
# #                        lambda:process_groups_dict['CCS'].get_electricity_consumption()/jet_fuel_gal_per_hr(),
# #                        'MW/gal'),))  


# # Material cost
# TEA_feeds = [i for i in sys.feeds if i.price]
# # TEA_products = [i for i in sys.products if i.price]

# def get_material_cost_breakdown():
#     group_material_costs = {}
#     for group in process_groups:
#         group_material_costs[group.name] = 0
#     counted_feeds =[]
#     for feed in TEA_feeds:
#         for group in process_groups:
#             if group.name != 'Facilities_no_hu_group':
#                 for unit in group.units:
#                     for instream in unit.ins:
#                         if instream.shares_flow_rate_with(feed) and not feed in counted_feeds:
#                             group_material_costs[group.name] += feed.price*feed.F_mass/jet_fuel_gal_per_hr()
#                             counted_feeds.append(feed)
#     group_material_costs['BT_group'] += BT.natural_gas_price*BT.natural_gas.F_mass/jet_fuel_gal_per_hr()
#     return group_material_costs

# def get_material_cost_breakdown_fractional():
#     mcb_dict = get_material_cost_breakdown()
#     sum_all = sum([v for k,v in mcb_dict.items()])
#     mcbf_dict = {}
#     for k,v in mcb_dict.items():
#         mcbf_dict[k] = mcb_dict[k]/sum_all
#     return mcbf_dict


# metrics.extend((Metric('Preprocessing_group - material cost',
#                        lambda:get_material_cost_breakdown()['Preprocessing_group'],
#                        '$/gal'),))

# metrics.extend((Metric('Pretreatment_group - material cost',
#                        lambda:get_material_cost_breakdown()['Pretreatment_group'],
#                        '$/gal'),))

# metrics.extend((Metric('Fermentation_group - material cost',
#                        lambda:get_material_cost_breakdown()['Fermentation_group'],
#                        '$/gal'),))

# metrics.extend((Metric('Upgrading_group - material cost',
#                        lambda:get_material_cost_breakdown()['Upgrading_group'],
#                        '$/gal'),))

# metrics.extend((Metric('WWT_group - material cost',
#                        lambda:get_material_cost_breakdown()['WWT_group'],
#                        '$/gal'),))

# metrics.extend((Metric('HXN_group - material cost',
#                        lambda:get_material_cost_breakdown()['HXN_group'],
#                        '$/gal'),))

# metrics.extend((Metric('BT_group - material cost',
#                        lambda:get_material_cost_breakdown()['BT_group'],
#                        '$/gal'),))

# metrics.extend((Metric('CT_group - material cost',
#                        lambda:get_material_cost_breakdown()['CT_group'],
#                        '$/gal'),))

# metrics.extend((Metric('Facilities_no_hu_group - material cost',
#                        lambda:get_material_cost_breakdown()['Facilities_no_hu_group'],
#                        '$/gal'),))
# # metrics.extend((Metric('CCS - material cost',
# #                        lambda:get_material_cost_breakdown()['CCS'],
# #                        '$/gal'),))

# 3. LCA
# in g CO2 eq / MJ blend fuel
_total_energy_per_hr = lambda: (jet_fuel.LHV + diesel.LHV + gasoline.LHV + ethanol_to_sold.LHV) / 1000 # from kJ to MJ
_total_energy_per_year = lambda: _total_energy_per_hr() * sys.operating_hours

_total_energy_per_hr_1 = lambda: (jet_fuel.LHV + diesel.LHV + gasoline.LHV + ethanol_to_sold.LHV + get_excess_power() * 3600) / 1000 # for energy allocation
_total_energy_per_year_1 = lambda: _total_energy_per_hr_1() * sys.operating_hours

_total_energy_to1 = lambda: _total_energy_per_year_1() / _total_energy_per_year()

main_product = [jet_fuel]
coproducts = [diesel, gasoline,ethanol_to_sold]
#impurities = [CH4_C2H6] # not counted here

emissions = [i for i in F.stream if i.source and not i.sink and i not in main_product and i not in coproducts]


# Carbon balance
total_C_in = sum([feed.get_atomic_flow('C') for feed in sys.feeds])
total_C_out = sum([i.get_atomic_flow('C') for i in emissions]) + sum([i.get_atomic_flow('C') for i in main_product]) +\
              sum([i.get_atomic_flow('C') for i in coproducts]) 
C_bal_error = (total_C_out - total_C_in)/total_C_in

# Feedstock contribution
get_GWP_feedstock_input = lambda: sys.get_material_impact(feedstock, key='GWP100') * 1000 / _total_energy_per_year()

# Non-biogenic emissions (ash)
get_GWP_missions_waste = lambda: sys.get_material_impact(F.ash, key='GWP100') * 1000 / _total_energy_per_year() 

# Non-biogenic emissions (enzyme+CSL)
get_GWP_emissions_C_source = lambda: (F.CSL.get_atomic_flow('C') + F.enzyme_M301.get_atomic_flow('C'))\
                                  * SAF_chemicals.CO2.MW * 1000 / _total_energy_per_hr()         
                                  
# Displacement GWP (ethnol_to_sold displacement, gasoline, diesel)
get_GWP_displacement = lambda: (sys.get_material_impact(F.ethanol_to_sold, key='GWP100') + sys.get_material_impact(F.gasoline, key='GWP100') + sys.get_material_impact(F.diesel, key='GWP100')) * 1000 / _total_energy_per_year() 
                                  
get_GWP_emissions_non_BT = lambda: get_GWP_missions_waste() + get_GWP_emissions_C_source()

# Non-biogenic missions (BT)
get_GWP_emissions_BT = lambda: F.natural_gas.get_atomic_flow('C') * SAF_chemicals.CO2.MW * 1000 / _total_energy_per_hr()

# Material contribution
get_GWP_material_total = lambda: sys.get_total_feeds_impact('GWP100') * 1000 / _total_energy_per_year()

get_GWP_NG = lambda: sys.get_material_impact(F.natural_gas, key='GWP100') * 1000 / _total_energy_per_year()

get_GWP_caustic = lambda: (sys.get_material_impact(F.NaOH, key='GWP100') + sys.get_material_impact(F.caustic, key='GWP100')) * 1000 / _total_energy_per_year()

get_GWP_other_materials = lambda: get_GWP_material_total()  - get_GWP_feedstock_input() - get_GWP_NG()

# Total = emission + material
get_GWP_total_displa = lambda: get_GWP_material_total() + get_GWP_emissions_BT() + get_GWP_emissions_non_BT() + get_GWP_displacement() # displacement method

get_GWP_total_hydrid = lambda: get_GWP_material_total() + get_GWP_emissions_BT() + get_GWP_emissions_non_BT() # hybrid method

get_GWP_total_energy_allocation = lambda: get_GWP_total_hydrid() / _total_energy_to1() # energy allocation method

# Electricity (BT satisfies all electricity in system by buying natural gas if needed, no buying electricity)
get_electricity_use_offset_total = lambda: sum(i.power_utility.rate for i in sys.units) # .= 0 per hour

get_GWP_electricity_use_total = lambda: get_electricity_use_offset_total() * GWP_CFs['electricity'] * 1000 / _total_energy_per_hr() # . = 0

get_electricity_demand_total = get_electricity_use = lambda: -BT.power_utility.rate # Outside BT + BT consumption 

get_electricity_demand_cooling = lambda: F.CT.power_utility.rate + F.CWP.power_utility.rate

get_electricity_demand_non_cooling = lambda: get_electricity_demand_total() - get_electricity_demand_cooling()

get_electricity_demand_cooling_frac = lambda: get_electricity_demand_cooling() / get_electricity_demand_total()

get_electricity_demand_non_cooling_frac = lambda: get_electricity_demand_non_cooling() / get_electricity_demand_total()


# Steam contribution
get_steam_heating = lambda: sum([i.duty for i in BT.steam_utilities]) # in kJ/hr

get_steam_electricity = lambda: 3600. * BT.electricity_demand / BT.turbogenerator_efficiency # in kJ/hr

get_steam_total = lambda: get_steam_heating() + get_steam_electricity()

get_steam_frac_heating = lambda: get_steam_heating() / get_steam_total()

get_steam_frac_electricity = lambda: get_steam_electricity() / get_steam_total()

get_steam_frac_cooling = lambda: get_steam_frac_electricity() * get_electricity_demand_cooling_frac()

get_steam_frac_non_cooling = lambda: get_steam_frac_electricity() * get_electricity_demand_non_cooling_frac()

# Steam GWP allocation

get_GWP_steam_total = lambda: get_GWP_NG() + get_GWP_emissions_BT()

get_GWP_heating_demand = lambda: get_steam_frac_heating() * get_GWP_steam_total()

get_GWP_cooling_demand = lambda: get_steam_frac_cooling() * get_GWP_steam_total() + \
                                 get_electricity_demand_cooling_frac() * get_GWP_electricity_use_total()
                                 
get_GWP_electricity_non_cooling = lambda: get_steam_frac_non_cooling() * get_GWP_steam_total() + \
                                          get_electricity_demand_non_cooling_frac() * get_GWP_electricity_use_total()

# GWP allocation (displacement method)
get_GWP_electricity_credit = lambda: -get_excess_power() * GWP_CFs['electricity'] * 1000 / _total_energy_per_year()

get_GWP_total_displa_with_eletricity_credit = lambda: get_GWP_total_displa() + get_GWP_electricity_credit()

get_GWP_total_hydrid_with_eletricity_credit = lambda: get_GWP_total_hydrid() + get_GWP_electricity_credit()

get_GWP_jet_displa = lambda: get_GWP_total_displa_with_eletricity_credit()

get_GWP_jet_hybrid = lambda: get_GWP_total_hydrid_with_eletricity_credit() * get_jet_energy_ratio() # in g CO2-eq/MJ jet

get_GWP_jet_energy = lambda: get_GWP_total_energy_allocation() * get_jet_energy_ratio_2()

# get_GWP_diesel = lambda: get_GWP_total_hydrid_with_eletricity_credit() * get_diesel_energy_ratio() / 100

# get_GWP_gasoline = lambda: get_GWP_total_hydrid_with_eletricity_credit() * get_gasoline_energy_ratio() / 100

get_SAF_abatement_cost = lambda: (get_jet_price_per_GJ() - kerosene_price_per_GJ) / (89-get_GWP_jet_displa()) * 1000 # in $/tonne CO2


metrics.extend((Metric('GWP - total', get_GWP_total_displa, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - electricity credit', get_GWP_electricity_credit, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - total-electricity credit', get_GWP_total_displa_with_eletricity_credit, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - feedstock', get_GWP_feedstock_input, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - NG', get_GWP_NG, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - other materials', get_GWP_other_materials, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - caustic', get_GWP_caustic, 'g CO2-eq/MJ blend fuel', 'LCA'),))

# metrics.extend((Metric('GWP - lime', get_GWP_lime, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - electricity', get_GWP_electricity_use_total, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - non biogenic emissions', get_GWP_emissions_BT, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - other non biogenic emissions', get_GWP_emissions_non_BT, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - heating_demand', get_GWP_heating_demand, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - cooling_demand', get_GWP_cooling_demand, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - electricity non cooling', get_GWP_electricity_non_cooling, 'g CO2-eq/MJ blend fuel', 'LCA'),))

metrics.extend((Metric('GWP - jet_displa', get_GWP_jet_displa, 'g CO2-eq/MJ jet fuel', 'LCA'),))

metrics.extend((Metric('GWP - jet_hybrid', get_GWP_jet_hybrid, 'g CO2-eq/MJ jet fuel', 'LCA'),))

metrics.extend((Metric('GWP - jet_energy', get_GWP_jet_energy, 'g CO2-eq/MJ jet fuel', 'LCA'),))

metrics.extend((Metric('GWP - displacement', get_GWP_displacement, 'g CO2-eq/MJ', 'LCA'),))

metrics.extend((Metric('GWP - SAF_abatement_cost', get_SAF_abatement_cost, '$/tonne CO2', 'LCA'),))

#%%

def create_model(system=sys,
                 metrics=metrics,
                 N=2000,
                 rule='L',
                 notify_runs=10,):
    model = Model(sys,metrics)
    param = model.parameter
    
    # ============================================================================
    # TEA parameters
    # ============================================================================
    ###### Ethanol distribution ######
    S304 = F.S304
    D = shape.Uniform(0.6,1.0)
    @param(name='Ethanol split for SAF', element=S304, kind='coupled', units='%',
            baseline=1.0, distribution=D)
    def set_ethanol_split(split):
        S304.split = split
    
    
    
    ##### Financial parameters #####
    D = shape.Triangle(1000000/24/0.8, 2000000/24/0.8, 3000000/24/0.8)
    @param(name='Feedstock flow', element='feedstock', kind='coupled', units='kg/hr',
            baseline=2000000/24/0.8, distribution=D)
    def set_feedstock_flow(flow):
        F.feedstock.F_mass = flow
    
    
    
    D = shape.Triangle(0.41/0.789*0.8, 0.41/0.789, 0.41/0.789*1.3)
    @param(name='Ethanol price', element='ethanol', kind='isolated', units='$/kg',
            baseline=0.41/0.789, distribution=D)
    def set_ethanol_price(price):
        F.ethanol_to_sold.price = price
        
        
    D = shape.Triangle(0.84, 0.9, 0.96)
    @param(name='Plant uptime', element='TEA', kind='isolated', units='%',
            baseline=0.9, distribution=D)
    def set_operating_days(uptime):
        tea_SAF.operating_days = 365. * uptime



    D = shape.Triangle(0.75, 1, 1.25)
    @param(name='TCI ratio', element='TEA', kind='isolated', units='% of baseline',
            baseline=1, distribution=D)
    def set_TCI_ratio(new_ratio):
        old_ratio = tea_SAF._TCI_ratio_cached
        for unit in sys.units:
            if hasattr(unit, 'cost_items'):
                for item in unit.cost_items:
                    unit.cost_items[item].cost /= old_ratio
                    unit.cost_items[item].cost *= new_ratio
        tea_SAF._TCI_ratio_cached = new_ratio



    ##### Material price #####
    D = shape.Triangle(0.0583, 0.0637, 0.069) # From historical price
    @param(name='Electricity price', element='Electricity', kind='isolated', units='$/kWh',
            baseline=0.0637, distribution=D)
    def set_electricity_price(price): 
        bst.PowerUtility.price = price



    natural_gas_price = price['natural gas']
    D = shape.Triangle(natural_gas_price*0.9, natural_gas_price, natural_gas_price*1.1)
    @param(name='Natural gas price', element='Natural gas', kind='isolated', units='$/kWh',
            baseline=natural_gas_price, distribution=D)
    def set_natural_gas_price(price): 
        F.natural_gas.price = price



    feedstock = F.feedstock
    moisture = F.feedstock.imass['Water'] / F.feedstock.F_mass
    D = shape.Triangle(75, 87.5, 100)
    @param(name='Feedstock unit price', element='Feedstock', kind='isolated', units='$/dry-ton',
            baseline=87.5, distribution=D)
    def set_feedstock_price(price):
        F.feedstock.price = price * (1-moisture) / 1000 # in $/kg



    ash_disposal_price = price['ash disposal']
    D = shape.Triangle(ash_disposal_price*1.5, ash_disposal_price, ash_disposal_price*0.5)
    @param(name='Ash disposal price', element='Ash disposal', kind='isolated', units='$/kg',
            baseline=ash_disposal_price, distribution=D)
    def set_ash_disposal_price(price):
        F.ash.price = price



    enzymeM301_price = price['enzyme']
    D = shape.Triangle(enzymeM301_price*0.8, enzymeM301_price, enzymeM301_price*1.2)
    @param(name='Enzyme_price', element='Enzyme', kind='isolated', units='$/kg',
            baseline=enzymeM301_price, distribution=D)
    def set_enzymeM301_price(price):
        F.enzyme_M301.price = price



    CSL_price = price['CSL']
    D = shape.Triangle(CSL_price*0.8, CSL_price, CSL_price*1.2)
    @param(name='CSL price', element='CSL', kind='isolated', units='$/kg',
            baseline=CSL_price, distribution=D)
    def set_CSL_price(price):
        F.CSL.price = price



    DAP_price = price['DAP']
    D = shape.Triangle(DAP_price*0.8, DAP_price, DAP_price*1.2)
    @param(name='DAP price', element='DAP', kind='isolated', units='$/kg',
            baseline=DAP_price, distribution=D)
    def set_DAP_price(price):
        F.DAP.price = price



    Syndol_catalyst_price = price['Syndol catalyst']
    D = shape.Triangle(Syndol_catalyst_price*0.7, Syndol_catalyst_price, Syndol_catalyst_price*1.3)
    @param(name='Syndol catalyst price', element='Syndol catalyst', kind='isolated', units='$/kg',
            baseline=Syndol_catalyst_price, distribution=D)
    def set_Syndol_catalyst_price(price):
        F.R401.catalyst_price = price



    Oligomerization1_catalyst_price = price['Ni-loaded aluminosilicate catalyst']
    D = shape.Triangle(Oligomerization1_catalyst_price*0.7, Oligomerization1_catalyst_price, Oligomerization1_catalyst_price*1.3)
    @param(name='Oligomerization1 catalyst price', element='Oligomerization1 catalyst', kind='isolated', units='$/kg',
            baseline=Oligomerization1_catalyst_price, distribution=D)
    def set_oligomerization1_catalyst_price(price):
        F.R402.catalyst_price = price



    Oligomerization2_catalyst_price = price['Aluminosilicate catalyst']
    D = shape.Triangle(Oligomerization2_catalyst_price*0.7, Oligomerization2_catalyst_price, Oligomerization2_catalyst_price*1.3)
    @param(name='Oligomerization2 catalyst price', element='Oligomerization2 catalyst', kind='isolated', units='$/kg',
            baseline=Oligomerization2_catalyst_price, distribution=D)
    def set_oligomerization2_catalyst_price(price):
        F.R403.catalyst_price = price



    Hydrogenation_catalyst_price = price['Como catalyst']
    D = shape.Triangle(Hydrogenation_catalyst_price*0.7, Hydrogenation_catalyst_price, Hydrogenation_catalyst_price*1.3)
    @param(name='Hydrogenation catalyst price', element='Hydrogenation catalyst', kind='isolated', units='$/kg',
            baseline=Hydrogenation_catalyst_price, distribution=D)
    def set_hydrogenation_catalyst_price(price):
        F.R404.catalyst_price = price



    H2_price = price['h2']
    D = shape.Triangle(H2_price*0.7, H2_price, H2_price*1.3)
    @param(name='H2 price', element='H2', kind='isolated', units='$/kg',
            baseline=H2_price, distribution=D)
    def set_hydrogen_price(price):
        F.hydrogen.price = price



    D = shape.Triangle(0.2/3.78/845*1000, 0.4/3.78/845*1000, 0.6/3.78/845*1000)
    @param(name='oc price', element='oc', kind='isolated', units='$/kg',
            baseline=0.4/3.78/845*1000, distribution=D)
    def set_oc_price(price):
        F.oc.price = price



    ###### Coproduct price ######
    diesel_price = price['diesel']
    D = shape.Triangle(diesel_price*0.7, diesel_price, diesel_price*1.5) # Fluctuation maximum rate=0.4
    @param(name='Diesel price', element='Diesel', kind='isolated', units='$/kg',
            baseline=diesel_price, distribution=D)
    def set_diesel_price(price):
        diesel.price = price



    gasoline_price = price['gasoline']
    D = shape.Triangle(gasoline_price*0.8, gasoline_price, gasoline_price*1.2) # Fluctuation maximum rate=0.2
    @param(name='Gasoline price', element='Gasoline', kind='isolated', units='$/kg',
            baseline=gasoline_price, distribution=D)
    def set_gasoline_price(price):
        gasoline.price = price



    # ###### Bagasse distribution ######
    # S102 = F.S102
    # D = shape.Uniform(0.5,1.0)
    # @param(name='Bagasse split for ethanol', element=S102, kind='coupled', units='%',
    #        baseline=0.8, distribution=D)
    # def set_bagasse_split(split):
    #     S102.split = split



    ###### Pretreatment parameters ######
    M201 = F.M201
    D = shape.Triangle(0.25, 0.305, 0.4)
    @param(name='Pretreatment solids loading', element=M201, kind='coupled', units='%',
            baseline=0.305, distribution=D)
    def set_pretreatment_solids_loading(loading):
        M201.solids_loading = loading



    R201 = F.R201
    D = shape.Triangle(0.06, 0.099, 0.12)
    @param(name='Pretreatment glucan-to-glucose', element=R201, kind='coupled', units='%',
            baseline=0.099, distribution=D)
    def set_R201_glucan_conversion(X): 
        R201.reactions[0].X = X    



    ###### Fermentation parameters ######
    M301 = F.M301
    R301 = F.R301

    D = shape.Triangle(0.175, 0.2, 0.25)
    @param(name='Enzymatic hydrolysis solids loading', element=M301, kind='coupled', units='%',
            baseline=0.2, distribution=D)
    def set_R301_solids_loading(loading):
        M301.solids_loading = loading



    D = shape.Triangle(10, 20, 30)
    @param(name='Enzyme loading', element=M301, kind='coupled', units='mg/g',
            baseline=20, distribution=D)
    def set_R301_enzyme_loading(loading):
        M301.enzyme_loading = loading



    D = shape.Triangle(0.002, 0.0025, 0.003)
    @param(name='CSL loading', element=R301, kind='coupled', units='g/L',
            baseline=0.0025, distribution=D)
    def set_CSL_loading(loading):
        R301.CSL_loading = loading



    D = shape.Triangle(0.75, 0.9, 0.948-1e-6)
    @param(name='Enzymatic hydrolysis glucan-to-glucose', element=R301, kind='coupled', units='%',
            baseline=0.9, distribution=D)
    def set_R301_glucan_to_glucose_conversion(X):
        R301.saccharification_rxns[3].X = X
           


    D = shape.Triangle(0.9, 0.95, 0.97)
    @param(name='Fermentation glucose-to-ethanol', element=R301, kind='coupled', units='%',
            baseline=0.95, distribution=D)
    def set_R301_glucose_to_ethanol_conversion(X):
        R301.cofermentation_rxns[0].X = X



    ##### Upgrading parameters #####
    R401 = F.R401
    R402 = F.R402
    R403 = F.R403
    R404 = F.R404



    D = shape.Uniform(0.995*0.988*0.9, 0.995*0.988)
    @param(name='Dehydration ethanol-to-ethylene', element=R401, kind='coupled', units='%',
            baseline=0.995*0.988, distribution=D)
    def set_R401_ethanol_conversion(X):
        R401.dehydration_rxns[0].X = X



    D = shape.Triangle(0.3, 0.43, 2)
    @param(name='Dehydration WHSV', element=R401, kind='coupled', units='h^-1',
            baseline=0.43, distribution=D)
    def set_R401_WHSV(X):
        R401.WHSV = X



    D = shape.Uniform(3.14/3600, 4.05/3600)
    @param(name='Dehydration residence time', element=R401, kind='coupled', units='hr',
            baseline=3.14/3600, distribution=D)
    def set_R401_residence_time(X):
        R401.tau = X


    D = shape.Uniform(7884*0.8,7884)
    @param(name='Dehydration catalyst longevity', element=R401, kind='coupled', units='hr',
            baseline=7884, distribution=D)
    def set_R401_catalyst_longevity(t):
        R401.catalyst_lifetime = t


    D = shape.Uniform(0.988*0.9, 0.988)
    @param(name='1st oligomerization ethylene-to-C4H8', element=R402, kind='coupled', units='%',
            baseline=0.988, distribution=D)
    def set_R402_ethylene_conversion(X):
        R402.oligomerization_rxns[0].X = X
        
        
        
    D = shape.Uniform(0.5, 5) 
    @param(name='1st oligomerization WHSV', element=R402, kind='coupled', units='h^-1',
            baseline=5, distribution=D)
    def set_R402_WHSV(X):
        R402.WHSX = X



    D = shape.Triangle(48*0.8, 48, 48*1.2) 
    @param(name='1st oligomerization residence time', element=R402, kind='coupled', units='hr',
            baseline=48, distribution=D)
    def set_R402_residence_time(X):
        R402.tau = X


    D = shape.Uniform(7884*0.8,7884)
    @param(name='1st oligomerization catalyst longevity', element=R402, kind='coupled', units='hr',
            baseline=7884, distribution=D)
    def set_R402_catalyst_longevity(t):
        R402.catalyst_lifetime = t



    D = shape.Uniform(0.5, 10) 
    @param(name='2nd oligomerization WHSV', element=R403, kind='coupled', units='h^-1',
            baseline=10, distribution=D)
    def set_R403_WHSV(X):
        R403.WHSX = X



    D = shape.Triangle(48*0.8, 48, 48*1.2) 
    @param(name='2nd oligomerization residence time', element=R403, kind='coupled', units='hr',
            baseline=48, distribution=D)
    def set_R403_residence_time(X):
        R403.tau = X



    D = shape.Uniform(7884*0.8,7884)
    @param(name='2nd oligomerization catalyst longevity', element=R403, kind='coupled', units='hr',
            baseline=7884, distribution=D)
    def set_R403_catalyst_longevity(t):
        R403.catalyst_lifetime = t



    D = shape.Uniform(1, 3)
    @param(name='Hydrogenation WHSV', element=R404, kind='coupled', units='h^-1',
            baseline=3, distribution=D)
    def set_R404_WHSV(X):
        R404.WHSX = X



    D = shape.Triangle(1*0.8, 1, 1*1.2) 
    @param(name='Hydrogenation residence time', element=R404, kind='coupled', units='hr',
            baseline=1, distribution=D)
    def set_R404_residence_time(X):
        R404.tau = X



    D = shape.Uniform(7884*0.8,7884)
    @param(name='Hydrogenation catalyst longevity', element=R404, kind='coupled', units='hr',
            baseline=7884, distribution=D)
    def set_R404_catalyst_longevity(t):
        R404.catalyst_lifetime = t



    ##### Facilities parameter #####
    BT = F.BT
    D = shape.Uniform(0.8*(1-0.1), 0.8*(1+0.1))
    @param(name='Boiler efficiency', element=BT, kind='coupled', units='',
            baseline=0.8, distribution=D)
    def set_boiler_efficiency(efficiency):
        BT.boiler_efficiency = efficiency



    # # =============================================================================
    # # LCA parameters
    # # =============================================================================
    D = shape.Uniform(-0.18363*0.8, -0.10874*0.8)
    @param(name='Feedstock GWP', element='Feedstock', kind='isolated', units='kg CO2-eq/kg',
            baseline=-0.14315*0.8, distribution=D)
    def set_feedstock_GWP(X):
        feedstock.characterization_factors['GWP100'] = X



    D = shape.Uniform(2.24*(1-0.5), 2.24*(1+0.5))
    @param(name='Enzyme GWP', element='Enzyme', kind='isolated', units='kg CO2-eq/kg',
            baseline=2.24, distribution=D)
    def set_enzyme_GWP(X):
        F.enzyme_M301.characterization_factors['GWP100'] = X



    D = shape.Uniform(1.55*(1-0.5), 1.55*(1+0.5))
    @param(name='CSL GWP', element='CSL', kind='isolated', units='kg CO2-eq/kg',
            baseline=1.55, distribution=D)
    def set_CSL_GWP(X):
        F.CSL.characterization_factors['GWP100'] = X



    D = shape.Uniform(1.6445*(1-0.5), 1.6445*(1+0.5))
    @param(name='DAP GWP', element='DAP', kind='isolated', units='kg CO2-eq/kg',
            baseline=1.6445, distribution=D)
    def set_DAP_GWP(X):
        F.DAP.characterization_factors['GWP100'] = X



    D = shape.Uniform(2.11*(1-0.5), 2.11*(1+0.5))
    @param(name='NaOH GWP', element='NaOH', kind='isolated', units='kg CO2-eq/kg',
            baseline=2.11, distribution=D)
    def set_NaOH_GWP(X):
        F.NaOH.characterization_factors['GWP100'] = X
        


    D = shape.Uniform(1.5624*(1-0.5), 1.5624*(1+0.5))
    @param(name='H2 GWP', element='H2', kind='isolated', units='kg CO2-eq/kg',
            baseline=1.5624, distribution=D)
    def set_H2_GWP(X):
        F.hydrogen.characterization_factors['GWP100'] = X



    D = shape.Uniform(0.4*(1-0.5), 0.4*(1+0.5))
    @param(name='Natural gas GWP', element='Natural gas', kind='isolated', units='kg CO2-eq/kWh',
            baseline=0.4, distribution=D)
    def set_natural_gas_GWP(X):
        F.natural_gas.characterization_factors['GWP100'] = X

    


    
    if N > 0:
        rule=rule
        np.random.seed(1234)
        samples=model.sample(N, rule)
        model.load_samples(samples)
        model.evaluate(notify=notify_runs)
        model.show()
        model.table.to_excel('model_table_repurposing.xlsx')
        df_rho,df_p = model.spearman_r()
        df_rho.to_excel('df_rho_repurposing.xlsx')
        df_p.to_excel('df_p_repurposing.xlsx')
    else:
        model.show()
    return model

#%%

# # MFSP Coutour data generating
# y_data = S304_split = np.linspace(0.6,1,9)
# x_data = feedflow = np.linspace(40000,140000,17)
# w_data = []
# def MFSP_at_x_and_y_1(x,y):
#     F.S304.split = y
#     F.feedstock.F_mass = x
#     sys.simulate()
#     MFSP = get_MPSP_per_gallon()
#     return(MFSP)

# for j in y_data:
#     w_data.append([])
#     for i in x_data:
#         try:
#             print(j, i, MFSP_at_x_and_y_1(i,j))
#             w_data[-1].append(MFSP_at_x_and_y_1(i,j))        
#         except:
#             print('Needs_interpolation')
#             w_data[-1].append(0)  

# #%%

# # CI Coutour data generating (can change to displacement, hybrid, energy)
# y_data = S304_split = np.round(np.linspace(0.6,1,9),2)
# x_data = feedflow_CI = np.round(np.linspace(-0.147,-0.087,13),3)
# w_data = []
# def CI_at_x_and_y_1(x,y):  
#     F.S304.split = y
#     F.feedstock.characterization_factors['GWP100']  = x
#     sys.simulate()
#     CI_jet = get_GWP_jet_energy()
#     return(CI_jet)

# for j in y_data:
#     w_data.append([])
#     for i in x_data:
#         try:
#             print(j, i, CI_at_x_and_y_1(i,j))
#             w_data[-1].append(CI_at_x_and_y_1(i,j))        
#         except:
#             print('Needs_interpolation')
#             w_data[-1].append(0)

# #%%

# Co-processing ratio vs MFSP
y_data = S304_split = np.arange(start=0.95,stop=1.1, step=0.01)
x_data = feedflow = np.arange(start=100000,stop=140001, step=100)
w_data = []
def MFSP_CR_at_x_and_y_1(x,y):  
    F.S304.split = y
    F.feedstock.F_mass = x
    sys.simulate()
    MFSP = get_MPSP_per_gallon() # in per gallon
    CR = get_coprocessing_ratio()
    return(MFSP,CR)

for j in y_data:
    w_data.append([])
    for i in x_data:
        try:
            print(j, i, MFSP_CR_at_x_and_y_1(i,j))
            w_data[-1].append(MFSP_CR_at_x_and_y_1(i,j))        
        except:
            print('Needs_interpolation')
            w_data[-1].append(0)
