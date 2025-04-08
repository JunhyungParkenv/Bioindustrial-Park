# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:53:46 2025

@author: Junhyung Park
"""
import numpy as np
import thermosteam as tmo
import biosteam as bst
import qsdsan as qs
from biorefineries.VFA._chemicals import chems

__all__ = ('load_preferences_and_process_settings', 'add_utility_agent', 'price', 'GWP_CFs', 'FEC_factors')


# =============================================================================
# System preferences and settings
# =============================================================================

def load_preferences_and_process_settings(T='K', flow_units='kg/hr', 
                                          N=4, P_units='1 atm', CE=541.7, 
                                          indicator='GWP100', electricity_price=0.07, 
                                          electricity_EI=0.48):
    """
    Load preferences and process settings for the VFA recovery system.
    Default values are used unless specified.
    """
    # --- Set BioSTEAM preferences ---
    bst.preferences.update(
        T=T,
        flow=flow_units,  # Default flow rate unit
        N=N,  # Decimal precision for outputs
        P=P_units,  # Default pressure with units (e.g., "1 atm")
        composition=False,  # Display stream compositions
    )
    bst.preferences.light_mode()  # Use light mode for reports
    bst.preferences.save()  # Save preferences globally
    
    # --- Set process economic settings ---
    bst.settings.CEPCI = CE  # CEPCI (Chemical Engineering Plant Cost Index)
    bst.settings.electricity_price = electricity_price  # Electricity price ($/kWh)

    # --- Set environmental impact indicators ---
    # Define GWP (Global Warming Potential) as the main impact category
    bst.settings.define_impact_indicator(key=indicator, units='kg*CO2e')
    bst.settings.set_electricity_CF(indicator, electricity_EI, 
                                    basis='kWhr', units='kg*CO2e')

    # --- Configure heating agents ---
    lps = bst.HeatUtility.get_heating_agent('low_pressure_steam')
    mps = bst.HeatUtility.get_heating_agent('medium_pressure_steam')
    hps = bst.HeatUtility.get_heating_agent('high_pressure_steam')

    # Update steam temperatures (in Kelvin)
    lps.T = 152 + 273.15  # Low-pressure steam
    mps.T = 233 + 273.15  # Medium-pressure steam
    hps.T = 266 + 273.15  # High-pressure steam

    # --- Configure cooling agents ---
    cooling_water = bst.HeatUtility.get_cooling_agent('cooling_water')
    cooling_water.T = 28 + 273.15  # Cooling water supply temperature
    cooling_water.T_limit = cooling_water.T + 9  # Maximum temperature increase
    cooling_water.regeneration_price = 0  # Regeneration cost for cooling water

    chilled_water = bst.HeatUtility.get_cooling_agent('chilled_water')
    chilled_water.T = 5 + 273.15  # Chilled water supply temperature
    chilled_water.heat_transfer_price = 0.00002  # Heat transfer cost ($/kWh)

    # --- Set cost parameters for heat agents ---
    for agent in (lps, mps, hps, cooling_water, chilled_water):
        agent.heat_transfer_price = agent.regeneration_price = 0

tmo.settings.set_thermo(chems)

# =============================================================================
# Utility Agents for Heating and Cooling
# =============================================================================

def add_utility_agent():
    """
    Add specialized heating and cooling agents for the VFA process.
    """
    # Add heating agent (e.g., Dowtherm A)
    DPO_chem = qs.Chemical('DPO', search_ID='101-84-8')
    BIP_chem = qs.Chemical('BIP', search_ID='92-52-4')
    DPO = qs.Component.from_chemical('DPO', chemical=DPO_chem)
    BIP = qs.Component.from_chemical('BIP', chemical=BIP_chem)
    HTF_thermo = bst.Thermo((DPO, BIP))
    HTF = bst.UtilityAgent('HTF', DPO=0.735, BIP=0.265, T=673.15, P=10.6 * 101325, phase='g',
                           thermo=HTF_thermo, regeneration_price=1)

    bst.HeatUtility.heating_agents.append(HTF)

    # Add low-temperature cooling agent
    DEC_chem = qs.Chemical('Decamethyltetrasiloxane', search_ID='141-62-8')
    OCT_chem = qs.Chemical('Octamethyltrisiloxane', search_ID='107-51-7')
    DEC = qs.Component.from_chemical('DEC', chemical=DEC_chem)
    OCT = qs.Component.from_chemical('OCT', chemical=OCT_chem)
    LTF_thermo = bst.Thermo((DEC, OCT))
    LTF = bst.UtilityAgent('LTF', DEC=0.4, OCT=0.6, T=173.15, P=5.2 * 101325, phase='l',
                           thermo=LTF_thermo, heat_transfer_price=0.00001317)

    bst.HeatUtility.cooling_agents.append(LTF)

# =============================================================================
# Prices for Techno-Economic Analysis (TEA)
# =============================================================================
price = {
    'water': 0.0002,  # Water price ($/kg)
    'electricity': 0.07,  # Electricity price ($/kWh)
    'AceticAcid': 1.5,  # Acetic acid price ($/kg)
    'PropionicAcid': 2.0,  # Propionic acid price ($/kg)
    'ButyricAcid': 2.5,  # Butyric acid price ($/kg)
    'ValericAcid': 3.0,  # Valeric acid price ($/kg)
    'LacticAcid': 1.8,  # Lactic acid price ($/kg)
    'AminoAcid': 5.0,  # Amino acid price ($/kg)
    'FattyAcid': 4.0,  # Fatty acid price ($/kg)
}

# =============================================================================
# GWP Factors for Life Cycle Assessment (LCA)
# =============================================================================
# IEM (CEM) Membrane, Nanofiltration Membrane, Ti mesh Current Collector, Carbon Cloth electrode
# Need to find NF (Polyamide-based), Frames (steel)
# GWP_CFs = {
#     'electricity': 0.48,  # Electricity GWP (kg CO2-eq/MJ) (1 kWh = 3.6 MJ)
#     'CEM': 2.0879,               # Membrane (Cation Exchange Membrane) (kg CO2-eq/kg) # Membrane Density=1200 kg/m3, Thickness=0.001m, 1.2kg/m2
#     'NF': 0.31,                # Membrane (NanoFiltration) (kg CO2-eq/kg) # paper ref. 0.12 g/m2
#     'Current Collector': 43.912, # Current Collector (kg CO2-eq/kg) # 4.5 g/cm3, 0.0001m -> 0.02 kg/m2
#     'Electrode' : 2.6292,         # Electrdoe (kg CO2-eq/kg) # 0.2 kg/m2
#     'Fe(CN)': 7.51281,           # Fe(CN) (kg CO2-eq/kg)
#     'Frames': 1.97,           # Support Frames (kg CO2-eq/kg) 1.97 kg CO2/kg * 50 kg/m2
# }
GWP_CFs = {
    'electricity': 0.449,  # Electricity (kg CO2-eq/kWh)
    'CEM': 2.5055,               # Membrane (Cation Exchange Membrane) (kg CO2-eq/m²)
    'NF': 0.001586,                # Membrane (NanoFiltration) (kg CO2-eq/m²), 0.12 g/m2
    'Current Collector': 0.8782, # Current Collector (Ti mesh) (kg CO2-eq/m²)
    'Electrode': 0.52584, # Electrode (Carbon Cloth) (kg CO2-eq/m²)
    'Fe(CN)': 0.3529,        # Fe(CN) (kg CO2-eq/m²) # 0.046875 g/cm2
    'Frames': 98.5,           # Support Frames (Steel) (kg CO2-eq/m²)
    'StainlessSteel': 48.0      # AC/DC 탱크에 사용되는 stainless steel (kg CO2-eq/kg), 8 kg/m3
}

# =============================================================================
# # FEC Impact Factors (kg oil eq per unit)
# =============================================================================
# Need to find NF (Polyamide-based), Frames (steel), AC tank (Stainless Steel)
# FEC_factors = {
#     'electricity': 0.037857143,  # kg oil eq per MJ (1 kWh = 3.6 MJ)
#     'CEM': 1.7775,          # kg oil eq per kg
#     'NF': 0.818,          # kg oil eq per kg, 36 MJ/kg / 44 MJ/kg (oil) 
#     'Electrode': 0.49897,        # kg oil eq per kg
#     'Current Collector': 11.399, # kg oil eq per kg
#     'Fe(CN)': 2.68771, # kg oil eq per kg
#     'Frames': 0.455, # kg oil eq per kg, 20 MJ/kg / 44 MJ/kg (oil) 
#     'StainlessSteel': 1.818  # AC/DC 탱크에 사용되는 stainless steel의 FEC (kg oil eq per kg), 80 MJ/kg / 44 MJ/kg (oil) 
# }
FEC_factors = {
    'electricity': 0.1363,  # (kg oil-eq/kWh)
    'CEM': 2.133,          # (kg oil-eq/m²)
    'NF': 0.9816,          # (kg oil-eq/m²)
    'Electrode': 0.0998,        # (kg oil-eq/m²)
    'Current Collector': 0.22798, # (kg oil-eq/m²)
    'Fe(CN)': 0.126, # (kg oil-eq/m²)
    'Frames': 22.75, # (kg oil-eq/m²), 20 MJ/kg / 44 MJ/kg (oil), 0.455 kg oil/kg * 50 kg/m2 = 22.75 kg oil/m2
    'StainlessSteel': 1.818  # (kg oil-eq/kg), AC/DC 탱크에 사용되는 stainless steel의 FEC, 80 MJ/kg / 44 MJ/kg (oil) 
}