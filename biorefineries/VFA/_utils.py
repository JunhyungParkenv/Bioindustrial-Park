#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:11:15 2020

This module is a modified implementation of modules from the following:
[1]	Bhagwat et al., Sustainable Production of Acrylic Acid via 3-Hydroxypropionic Acid from Lignocellulosic Biomass. ACS Sustainable Chem. Eng. 2021, 9 (49), 16659–16669. https://doi.org/10.1021/acssuschemeng.1c05441
[2]	Li et al., Sustainable Lactic Acid Production from Lignocellulosic Biomass. ACS Sustainable Chem. Eng. 2021, 9 (3), 1341–1351. https://doi.org/10.1021/acssuschemeng.0c08055
[3]	Cortes-Peña et al., BioSTEAM: A Fast and Flexible Platform for the Design, Simulation, and Techno-Economic Analysis of Biorefineries under Uncertainty. ACS Sustainable Chem. Eng. 2020, 8 (8), 3302–3310. https://doi.org/10.1021/acssuschemeng.9b07040

@author: sarangbhagwat
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VFA Production and TEA Utilities
Author: Customized for VFA Analysis
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import thermosteam as tmo
from biorefineries.VFA._chemicals import chems, chemical_groups, get_grouped_chemicals  # _chemicals 모듈 가져오기

__all__ = (
    "calculate_split_ratios",
    "calculate_vfa_concentration",
    "get_feedstock_flow",
    "calculate_MPSP",
    "adjust_recycle_flow",
    "compute_extra_chemical",
    "get_vfa_properties",
    "chems",
    "chemical_groups",
    "get_grouped_chemicals",
)

# =============================================================================
# Function to calculate split ratios for VFA separation
# =============================================================================

def calculate_split_ratios(chemical_ids, flow_stream1, flow_stream2):
    """
    Calculate split ratios for chemicals between two streams.
    Args:
        chemical_ids (list): List of chemical IDs.
        flow_stream1 (array): Flow rates in stream 1.
        flow_stream2 (array): Flow rates in stream 2.
    Returns:
        dict: Split ratios for each chemical.
    """
    flow1 = np.asarray(flow_stream1) + 1e-6  # Avoid division by zero
    flow2 = np.asarray(flow_stream2) + 1e-6
    splits = flow1 / (flow1 + flow2)
    return dict(zip(chemical_ids, splits))


# =============================================================================
# Function to calculate total VFA concentration in a stream
# =============================================================================

def calculate_vfa_concentration(stream, vfa_ids):
    """
    Calculate the total VFA concentration in a stream.
    Args:
        stream (thermosteam.Stream): The stream containing VFAs.
        vfa_ids (list): List of VFA chemical IDs (e.g., ['AceticAcid', 'ButyricAcid']).
    Returns:
        float: Total VFA concentration [g/L].
    """
    total_vfa_mass = sum(stream.imass[vfa] for vfa in vfa_ids)
    return total_vfa_mass / stream.F_vol  # g/L


# =============================================================================
# Function to calculate feedstock flow
# =============================================================================

def get_feedstock_flow(dry_composition, moisture_content, dry_flow_rate):
    """
    Calculate feedstock flow based on dry composition and moisture content.
    Args:
        dry_composition (dict): Dry chemical composition (e.g., {'Glucan': 0.35}).
        moisture_content (float): Fractional moisture content (e.g., 0.2 for 20%).
        dry_flow_rate (float): Dry feedstock flow rate [kg/hr].
    Returns:
        thermosteam.array: Feedstock flow array including water.
    """
    dry_array = tmo.Chemicals.kwarray(dry_composition)
    wet_flow = dry_flow_rate / (1 - moisture_content)
    moisture_array = tmo.Chemicals.kwarray({"Water": moisture_content})
    return wet_flow * (dry_array * (1 - moisture_content) + moisture_array)


# =============================================================================
# Function to calculate Minimum Product Selling Price (MPSP)
# =============================================================================

def calculate_MPSP(
    target_irr, selling_price, annual_sales_mass, project_duration, initial_investment, npv=0
):
    """
    Calculate the Minimum Product Selling Price (MPSP) at a given IRR.
    Args:
        target_irr (float): Target internal rate of return (e.g., 0.10 for 10%).
        selling_price (float): Initial product selling price [USD/kg].
        annual_sales_mass (float): Annual sales mass [kg/year].
        project_duration (int): Project duration [years].
        initial_investment (float): Initial capital investment [USD].
        npv (float): Net present value at initial selling price [USD] (default: 0).
    Returns:
        float: Minimum product selling price [USD/kg].
    """
    cash_flow_no_sales = (
        annual_sales_mass * selling_price
        - (npv + initial_investment)
        / sum(1 / ((1 + target_irr) ** i) for i in range(1, project_duration + 1))
    )
    return round(
        (cash_flow_no_sales + initial_investment / sum(1 / ((1 + target_irr) ** i) for i in range(1, project_duration + 1)))
        / annual_sales_mass,
        3,
    )


# =============================================================================
# Function to adjust recycle flow for specific reactant ratios
# =============================================================================

def adjust_recycle_flow(feed, recycle, reactant_ids, adjust_chemical_id, ratios):
    """
    Adjust the recycle flow to maintain desired reactant-to-chemical ratio.
    Args:
        feed (thermosteam.Stream): Feed stream.
        recycle (thermosteam.Stream): Recycle stream.
        reactant_ids (list): List of reactant chemical IDs.
        adjust_chemical_id (str): Chemical ID to adjust.
        ratios (list): Desired ratios of reactants to adjust chemical.
    Returns:
        tuple: Effluent stream and discarded recycle stream.
    """
    feed_chemical_needed = sum(
        feed.imol[reactant] * ratio for reactant, ratio in zip(reactant_ids, ratios)
    ) - feed.imol[adjust_chemical_id]
    recycle_chemical_extra = recycle.imol[adjust_chemical_id] - sum(
        recycle.imol[reactant] * ratio for reactant, ratio in zip(reactant_ids, ratios)
    )
    split_ratio = feed_chemical_needed / recycle_chemical_extra

    # Adjust recycle streams
    recycle_recycled = recycle.copy()
    recycle_recycled.mol *= split_ratio
    recycle_discarded = recycle.copy()
    recycle_discarded.mol *= 1 - split_ratio
    effluent = feed.copy()
    effluent.mix_from([feed, recycle_recycled])
    return effluent, recycle_discarded


# =============================================================================
# Function to compute extra chemical amount
# =============================================================================

def compute_extra_chemical(feed, recycle, reactants_ids, chemical_id, ratios):
    """
    Compute the extra amount of chemical required for reactions.
    Args:
        feed (thermosteam.Stream): Feed stream.
        recycle (thermosteam.Stream): Recycle stream.
        reactants_ids (list): List of reactant chemical IDs.
        chemical_id (str): ID of the chemical to calculate.
        ratios (list): Desired ratios of reactants to the chemical.
    Returns:
        float: Extra chemical amount required.
    """
    feed_reactants = feed.imol[reactants_ids]
    recycle_reactants = recycle.imol[reactants_ids]
    required_chemical = sum((ratios * (feed_reactants + recycle_reactants)))
    available_chemical = feed.imol[chemical_id] + recycle.imol[chemical_id]
    return available_chemical - required_chemical


# =============================================================================
# Function to output VFA chemical properties
# =============================================================================

def get_vfa_properties(vfa_ids, thermo, T=298.15, P=101325):
    """
    Get VFA properties such as MW, boiling point, etc.
    Args:
        vfa_ids (list): List of VFA chemical IDs.
        thermo: Thermosteam settings object.
        T (float): Temperature [K].
        P (float): Pressure [Pa].
    Returns:
        pd.DataFrame: DataFrame with VFA properties.
    """
    chemicals = thermo.chemicals
    data = []
    for vfa in vfa_ids:
        chem = chemicals[vfa]
        data.append({
            "ID": vfa,
            "MW": chem.MW,
            "BoilingPoint": chem.Tb,
            "Density": chem.rho,
            "Phase": chem.phase_ref,
        })
    return pd.DataFrame(data)


# =============================================================================
# Example usage of chemical_groups
# =============================================================================
# Usage example for group flow rate summaries:
# stream = tmo.Stream("example", AceticAcid=10, Water=90)
# grouped_flows = get_grouped_chemicals(stream)
# print(grouped_flows)

