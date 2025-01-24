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

__all__ = ('load_preferences_and_process_settings', 'add_utility_agent', 'price', 'GWP_CFs')


# =============================================================================
# System preferences and settings
# =============================================================================

def load_preferences_and_process_settings():
    """
    Load preferences and process settings for the VFA recovery system.
    Sets default thermodynamic, economic, and environmental impact parameters.
    """
    # Set BioSTEAM preferences
    bst.preferences.T = 298.15  # Default temperature (25°C)
    bst.preferences.flow = 'kg/hr'  # Default flow rate unit
    bst.preferences.N = 4  # Decimal precision
    bst.preferences.P = 'atm'  # Default pressure unit
    bst.preferences.composition = True
    bst.preferences.light_mode()
    bst.preferences.save()

    # Set process economic and LCA indicators
    bst.settings.CEPCI = 541.7  # CEPCI (Chemical Engineering Plant Cost Index, 2016 baseline)
    bst.settings.define_impact_indicator(key='GWP', units='kg CO2-eq')
    bst.settings.electricity_price = price['electricity']  # Electricity price
    bst.settings.set_electricity_CF('GWP', GWP_CFs['electricity'], basis='kWhr', units='kg CO2-eq')

    # Define heating agents
    lps = bst.HeatUtility.get_heating_agent('low_pressure_steam')
    mps = bst.HeatUtility.get_heating_agent('medium_pressure_steam')
    hps = bst.HeatUtility.get_heating_agent('high_pressure_steam')

    mps.T = 233 + 273.15  # Medium-pressure steam temperature
    hps.T = 266 + 273.15  # High-pressure steam temperature

    # Define cooling agents
    cooling_water = bst.HeatUtility.get_cooling_agent('cooling_water')
    cooling_water.T = 28 + 273.15  # Cooling water supply temperature
    cooling_water.T_limit = cooling_water.T + 9  # Max temperature increase
    cooling_water.regeneration_price = 0  # Regeneration cost

    chilled_water = bst.HeatUtility.get_cooling_agent('chilled_water')
    chilled_water.heat_transfer_price = 0  # Cost of chilled water

    # Set prices for heat transfer
    for agent in (lps, mps, hps, cooling_water, chilled_water):
        agent.heat_transfer_price = agent.regeneration_price = 0

    # Set thermo for the simulation
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
    'H2': 1.07,  # Hydrogen price ($/kg)
    'NaOH': 0.86,  # Sodium hydroxide price ($/kg)
    'AceticAcid': 1.5,  # Acetic acid price ($/kg)
    'PropionicAcid': 2.0,  # Propionic acid price ($/kg)
    'ButyricAcid': 2.5,  # Butyric acid price ($/kg)
    'ValericAcid': 3.0,  # Valeric acid price ($/kg)
    'HexanoicAcid': 3.5,  # Hexanoic acid price ($/kg)
    'LacticAcid': 1.8,  # Lactic acid price ($/kg)
    'AminoAcid': 5.0,  # Amino acid price ($/kg)
    'FattyAcid': 4.0,  # Fatty acid price ($/kg)
}


# =============================================================================
# GWP Factors for Life Cycle Assessment (LCA)
# =============================================================================
GWP_CFs = {
    'water': 0.0002,  # Water GWP (kg CO2-eq/kg)
    'electricity': 0.48,  # Electricity GWP (kg CO2-eq/kWh)
    'H2': 1.6,  # Hydrogen GWP
    'NaOH': 2.11,  # Sodium hydroxide GWP
    'AceticAcid': -0.5,  # Co-product credit
    'PropionicAcid': -0.7,  # Co-product credit
    'ButyricAcid': -0.8,  # Co-product credit
    'ValericAcid': -1.0,  # Co-product credit
    'HexanoicAcid': -1.2,  # Co-product credit
    'LacticAcid': 1.5,  # Lactic acid GWP
}

# =============================================================================
# Load Process Settings
# =============================================================================
load_preferences_and_process_settings()
