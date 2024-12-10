# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:44:05 2024

@author: Junhyung Park
"""
# %%  

# =============================================================================
# Setup
# =============================================================================
import thermosteam as tmo
from thermosteam import functional as fn
import pandas as pd

__all__ = ('chems', 'chemical_groups', 'get_grouped_chemicals')

# Constants
_cal2joule = 4.184

# Create a chemicals object
chems = tmo.Chemicals([])

# Dictionaries for tracking chemicals
database_chemicals_dict = {}
copied_chemicals_dict = {}
defined_chemicals_dict = {}

def chemical_database(ID, search_ID=None, phase=None, **kwargs):
    """Add a chemical from the Thermosteam database."""
    chemical = tmo.Chemical(ID, search_ID=search_ID, **kwargs)
    if phase:
        chemical.at_state(phase)
        chemical.phase_ref = phase
    chems.append(chemical)
    database_chemicals_dict[ID] = f'{ID}: {chemical.formula}/{chemical.MW}'
    return chemical

def chemical_copied(ID, ref_chemical, **data):
    """Copy an existing chemical and modify properties."""
    chemical = ref_chemical.copy(ID)
    chems.append(chemical)
    for attr, value in data.items():
        setattr(chemical, attr, value)
    copied_chemicals_dict[ID] = f'{ID}: {chemical.formula}/{chemical.MW}'
    return chemical

def chemical_defined(ID, **kwargs):
    """Define a new chemical with specified properties."""
    chemical = tmo.Chemical.blank(ID, **kwargs)
    chems.append(chemical)
    defined_chemicals_dict[ID] = f'{ID}: {chemical.formula}/{chemical.MW}'
    return chemical
#%%
# =============================================================================
# Create chemical objects available in database
# =============================================================================

H2O = chemical_database('H2O')

# =============================================================================
# Gases
# =============================================================================
H2 = chemical_database('H2', Hf=0)
CH4 = chemical_database('Methane')
CO = chemical_database('CarbonMonoxide', Hf=-26400 * _cal2joule)
CO2 = chemical_database('CO2')
O2 = chemical_database('O2')
N2 = chemical_database('N2')

# =============================================================================
# Soluble inorganics
# =============================================================================
NaCl = chemical_database('NaCl')
H2SO4 = chemical_database('H2SO4')
NaOH = chemical_database('NaOH')

# =============================================================================
# Soluble organics
# =============================================================================
AceticAcid = chemical_database('AceticAcid')  # C2
PropionicAcid = chemical_database('PropionicAcid')  # C3
ButyricAcid = chemical_database('ButyricAcid')  # C4
ValericAcid = chemical_database('ValericAcid')  # C5
HexanoicAcid = chemical_database('HexanoicAcid', search_ID='CaproicAcid')  # C6
LacticAcid = chemical_database('LacticAcid')  # C3
Ethanol = chemical_database('Ethanol')

# Additional defined chemicals
AminoAcid = chemical_database('AminoAcid', search_ID='Glycine')  # Representative amino acid
FattyAcid = chemical_database('FattyAcid', search_ID='PalmiticAcid')  # Representative fatty acid
Glucose = chemical_database('Glucose')
# %% Chemical groups for categorization
chemical_groups = {
    'Gases': ('H2', 'CH4', 'CO', 'CO2', 'O2', 'N2'),
    'Inorganics': ('H2SO4', 'NaOH', 'NaCl'),
    'VFAs': ('AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid', 'HexanoicAcid'),
    'Alcohols': ('Ethanol',),
    'OtherOrganics': ('LacticAcid', 'AminoAcid', 'FattyAcid'),
    'Sugars': ('Glucose',),
}

def get_grouped_chemicals(stream, units='kmol/hr'):
    """Summarize flow rates of chemicals in groups."""
    grouped_data = {}
    for group, IDs in chemical_groups.items():
        grouped_data[group] = stream.get_flow(units, IDs).sum()
    return pd.Series(grouped_data)

# %% Finalize and compile the chemicals
chems.compile()
# =============================================================================
chems.set_synonym('H2O', 'Water')
chems.set_synonym('H2SO4', 'SulfuricAcid')
chems.set_synonym('NaOH', 'SodiumHydroxide')
chems.set_synonym('CO2', 'CarbonDioxide')
chems.set_synonym('CO', 'CarbonMonoxide')
chems.set_synonym('O2', 'Oxygen')
# %% Finalize and compile the chemicals
tmo.settings.set_thermo(chems)