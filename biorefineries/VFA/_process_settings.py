# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:53:46 2025

@author: Junhyung Park
"""
import thermosteam as tmo
import biosteam as bst
from biorefineries.VFA._chemicals import chems

__all__ = ('load_process_settings', 'price', 'GWP_CFs')

# =============================================================================
# System preferences and settings
# =============================================================================

def load_process_settings():
    """
    Load process settings for VFA recovery system.
    Sets default thermodynamic, economic, and environmental impact parameters.
    """
    # Set the thermodynamic and economic baseline
    bst.CE = 541.7  # CEPCI index (2016 baseline)
    bst.PowerUtility.price = price['Electricity']  # Set electricity price

    # Define heating agents
    lps = bst.HeatUtility.get_heating_agent('low_pressure_steam')
    mps = bst.HeatUtility.get_heating_agent('medium_pressure_steam')
    hps = bst.HeatUtility.get_heating_agent('high_pressure_steam')

    mps.T = 233 + 273.15  # Medium-pressure steam temperature
    hps.T = 266 + 273.15  # High-pressure steam temperature

    # Define cooling agents
    cooling_water = bst.HeatUtility.get_cooling_agent('cooling_water')
    cooling_water.regeneration_price = 0  # Cooling water regeneration cost
    cooling_water.T = 28 + 273.15  # Cooling water supply temperature
    cooling_water.T_limit = cooling_water.T + 9  # Maximum temperature increase

    chilled_water = bst.HeatUtility.get_cooling_agent('chilled_water')
    chilled_water.heat_transfer_price = 0  # Chilled water cost

    # Set heat transfer and regeneration prices to zero for consistency
    for agent in (lps, mps, hps, cooling_water, chilled_water):
        agent.heat_transfer_price = agent.regeneration_price = 0

    # Set thermo for the simulation
    tmo.settings.set_thermo(chems)


# =============================================================================
# Prices for techno-economic analysis (TEA)
# =============================================================================
# Prices in 2023 USD/kg
price = {
    'Water': 0.0002,  # Cooling water price ($/kg)
    'Electricity': 0.07,  # Electricity price ($/kWh)
    'H2': 1.5,  # Hydrogen price ($/kg)
    'CH4': 0.3,  # Methane price ($/kg)
    'CO2': 0.01,  # Carbon dioxide ($/kg)
    'O2': 0.1,  # Oxygen price ($/kg)
    'N2': 0.05,  # Nitrogen price ($/kg)
    'NaCl': 0.02,  # Sodium chloride price ($/kg)
    'H2SO4': 0.043,  # Sulfuric acid price ($/kg)
    'NaOH': 0.2384,  # Sodium hydroxide price ($/kg)
    'AceticAcid': 1.5,  # Acetic acid price ($/kg)
    'PropionicAcid': 2.0,  # Propionic acid price ($/kg)
    'ButyricAcid': 2.5,  # Butyric acid price ($/kg)
    'ValericAcid': 3.0,  # Valeric acid price ($/kg)
    'HexanoicAcid': 3.5,  # Hexanoic acid price ($/kg)
    'LacticAcid': 1.8,  # Lactic acid price ($/kg)
    'Ethanol': 0.5,  # Ethanol price ($/kg)
    'Glucose': 0.3,  # Glucose price ($/kg)
    'AminoAcid': 5.0,  # Representative amino acid price ($/kg)
    'FattyAcid': 4.0,  # Representative fatty acid price ($/kg)
}

# =============================================================================
# Characterization factors (CFs) for life cycle analysis (LCA)
# =============================================================================
# Global warming potential (GWP) in kg CO2-eq/kg
GWP_CFs = {
    'Water': 0.0002,  # kg CO2-eq/kg water
    'Electricity': 0.48,  # kg CO2-eq/kWh
    'H2': 1.6,  # Hydrogen
    'CH4': 0.4,  # Methane
    'CO2': 0.01,  # Captured CO2 credit
    'O2': 0.1,  # Oxygen
    'N2': 0.05,  # Nitrogen
    'NaCl': 0.2,  # Sodium chloride
    'H2SO4': 0.2,  # Sulfuric acid
    'NaOH': 2.0,  # Sodium hydroxide
    'AceticAcid': -0.5,  # Co-product credit for acetic acid
    'PropionicAcid': -0.7,  # Co-product credit for propionic acid
    'ButyricAcid': -0.8,  # Co-product credit for butyric acid
    'ValericAcid': -1.0,  # Co-product credit for valeric acid
    'HexanoicAcid': -1.2,  # Co-product credit for hexanoic acid
    'LacticAcid': 1.5,  # Lactic acid production
    'Ethanol': -0.8,  # Co-product credit for ethanol
    'Glucose': 0.3,  # Glucose
    'AminoAcid': 2.0,  # Representative amino acid
    'FattyAcid': 1.8,  # Representative fatty acid
}

# =============================================================================
# Load settings
# =============================================================================
load_process_settings()