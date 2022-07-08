# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:44:17 2021

@author: yrc2
"""
from biosteam import MockFeature

(set_crushing_mill_oil_recovery, set_saccharification_oil_recovery, 
 set_cane_operating_days, set_sorghum_operating_days, 
 set_plant_capacity, 
 set_ethanol_price, set_biodiesel_price, set_natural_gas_price, set_electricity_price, 
 set_IRR, set_crude_glycerol_price, set_pure_glycerol_price,
 set_saccharification_reaction_time, set_cellulase_price, 
 set_cellulase_loading, set_reactor_base_cost,
 set_cane_glucose_yield, set_cane_xylose_yield, 
 set_sorghum_glucose_yield, set_sorghum_xylose_yield, 
 set_glucose_to_ethanol_yield, set_xylose_to_ethanol_yield,
 set_cofermentation_titer, set_cofermentation_productivity,
 set_cane_PL_content, set_sorghum_PL_content, set_cane_FFA_content,
 set_sorghum_FFA_content,  set_cane_oil_content, set_relative_sorghum_oil_content,
 set_TAG_to_FFA_conversion, set_oilcane_GWP, set_methanol_GWP, 
 set_pure_glycerine_GWP, set_cellulase_GWP, set_natural_gas_GWP
 ) = all_parameter_mockups = (
    MockFeature('Crushing mill oil recovery', '%', 'biorefinery'),
    MockFeature('Saccharification oil recovery', '%', 'biorefinery'),
    MockFeature('Cane operating days', 'day/yr', 'biorefinery'),
    MockFeature('Sorghum operating days', 'day/yr', 'biorefinery'),
    MockFeature('Annual crushing capacity', 'MT/yr', 'biorefinery'),
    MockFeature('Price', 'USD/L', 'Stream-ethanol'),
    MockFeature('Price', 'USD/L', 'Stream-biodiesel'),
    MockFeature('Price', 'USD/m3', 'Stream-natural gas'),
    MockFeature('Electricity price', 'USD/kWhr', 'biorefinery'),
    MockFeature('IRR', '%', 'biorefinery'),
    MockFeature('Price', 'USD/kg', 'Stream-crude glycerol'),
    MockFeature('Price', 'USD/kg', 'Stream-pure glycerine'),
    MockFeature('Reaction time', 'hr', 'Saccharification'),
    MockFeature('Price', 'USD/kg', 'cellulase'),
    MockFeature('Cellulase loading', 'wt. % cellulose', 'cellulase'),
    MockFeature('Base cost', 'million USD', 'Pretreatment reactor system'),
    MockFeature('Cane glucose yield', '%', 'Pretreatment and saccharification'),
    MockFeature('Sorghum glucose yield', '%', 'Pretreatment and saccharification'),
    MockFeature('Cane xylose yield', '%', 'Pretreatment and saccharification'),
    MockFeature('Sorghum xylose yield', '%', 'Pretreatment and saccharification'),
    MockFeature('Glucose to ethanol yield', '%', 'Cofermenation'),
    MockFeature('Xylose to ethanol yield', '%', 'Cofermenation'),
    MockFeature('Titer', 'g/L', 'Cofermentation'),
    MockFeature('Productivity', 'g/L', 'Cofermentation'),
    MockFeature('Cane PL content', '% oil', 'oilcane'),
    MockFeature('Sorghum PL content', '% oil', 'oilsorghum'),
    MockFeature('Cane FFA content', '% oil', 'oilcane'),
    MockFeature('Sorghum FFA content', '% oil', 'oilsorghum'),
    MockFeature('Cane oil content', 'dry wt. %', 'oilcane'),
    MockFeature('Relative sorghum oil content', 'dry wt. %', 'oilsorghum'),
    MockFeature('TAG to FFA conversion', '% oil', 'biorefinery'),
    MockFeature('GWP', 'kg*CO2-eq/kg', 'Stream-oilcane'),
    MockFeature('GWP', 'kg*CO2-eq/kg', 'Stream-methanol'),
    MockFeature('GWP', 'kg*CO2-eq/kg', 'Stream-pure glycerine'),
    MockFeature('GWP', 'kg*CO2-eq/kg', 'Stream-cellulase'),
    MockFeature('GWP', 'kg*CO2-eq/kg', 'Stream-natural gas'),
)
     
(MFPP, feedstock_consumption, biodiesel_production, ethanol_production, 
 electricity_production, natural_gas_consumption, TCI, 
 heat_exchanger_network_error, GWP_economic, GWP_ethanol, GWP_biodiesel, 
 GWP_crude_glycerol, GWP_electricity, GWP_ethanol_displacement,
 GWP_biofuel_allocation, GWP_ethanol_allocation,
 GWP_biodiesel_allocation, GWP_crude_glycerol_allocation,
 MFPP_derivative, 
 biodiesel_production_derivative, ethanol_production_derivative, 
 electricity_production_derivative, natural_gas_consumption_derivative, 
 TCI_derivative, GWP_economic_derivative, 
 GWP_ethanol_derivative, GWP_biodiesel_derivative,
 GWP_crude_glycerol_derivative, GWP_electricity_derivative
 ) = all_metric_mockups = (
    MockFeature('MFPP', 'USD/MT', 'Biorefinery'),
    MockFeature('Feedstock consumption', 'MT/yr', 'Biorefinery'),
    MockFeature('Biodiesel production', 'L/MT', 'Biorefinery'),
    MockFeature('Ethanol production', 'L/MT', 'Biorefinery'),
    MockFeature('Electricity production', 'kWhr/MT', 'Biorefinery'),
    MockFeature('Natural gas consumption', 'm3/MT', 'Biorefinery'),
    MockFeature('TCI', '10^6*USD', 'Biorefinery'),
    MockFeature('Heat exchanger network error', '%', 'Biorefinery'),
    MockFeature('GWP', 'kg*CO2*eq / USD', 'Economic allocation'),
    MockFeature('Ethanol GWP', 'kg*CO2*eq / L', 'Economic allocation'),
    MockFeature('Biodiesel GWP', 'kg*CO2*eq / L', 'Economic allocation'),
    MockFeature('Crude glycerol GWP', 'kg*CO2*eq / kg', 'Economic allocation'),
    MockFeature('Electricity GWP', 'kg*CO2*eq / MWhr', 'Economic allocation'),
    MockFeature('Ethanol GWP', 'kg*CO2*eq / L', 'Displacement allocation'),
    MockFeature('Biofuel GWP', 'kg*CO2*eq / GGE', 'Energy allocation'),
    MockFeature('Ethanol GWP', 'kg*CO2*eq / L', 'Energy allocation'),
    MockFeature('Biodiesel GWP', 'kg*CO2*eq / L', 'Energy allocation'),
    MockFeature('Crude-glycerol GWP', 'kg*CO2*eq / kg', 'Energy allocation'),
    MockFeature('MFPP derivative', 'USD/MT', 'Biorefinery'),
    MockFeature('Biodiesel production derivative', 'L/MT', 'Biorefinery'),
    MockFeature('Ethanol production derivative', 'L/MT', 'Biorefinery'),
    MockFeature('Electricity production derivative', 'kWhr/MT', 'Biorefinery'),
    MockFeature('Natural gas consumption derivative', 'cf/MT', 'Biorefinery'),
    MockFeature('TCI derivative', '10^6*USD', 'Biorefinery'),
    MockFeature('GWP derivative', 'kg*CO2*eq / USD', 'Economic allocation'),
    MockFeature('Ethanol GWP derivative', 'kg*CO2*eq / L', 'Ethanol'),
    MockFeature('Biodiesel GWP derivative', 'kg*CO2*eq / L', 'Biodiesel'),
    MockFeature('Crude glycerol GWP derivative', 'kg*CO2*eq / kg', 'Crude glycerol'),
    MockFeature('Electricity GWP derivative', 'kg*CO2*eq / MWhr', 'Electricity'),
)

tea_monte_carlo_metric_mockups = (
    MFPP, 
    TCI,
    ethanol_production,
    biodiesel_production,
    electricity_production,
    natural_gas_consumption
)

tea_monte_carlo_derivative_metric_mockups = (
    MFPP_derivative, 
    TCI_derivative,
    ethanol_production_derivative,
    biodiesel_production_derivative,
    electricity_production_derivative,
    natural_gas_consumption_derivative,
)

lca_monte_carlo_metric_mockups = (
    GWP_economic,
    GWP_ethanol,
    GWP_biodiesel,
    GWP_electricity,
    GWP_crude_glycerol,
)

lca_monte_carlo_derivative_metric_mockups = (
    GWP_economic_derivative,
    GWP_ethanol_derivative,
    GWP_biodiesel_derivative, 
    GWP_electricity_derivative,
    GWP_crude_glycerol_derivative,
)