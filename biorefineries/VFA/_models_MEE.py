# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:15:09 2025

@author: Junhyung Park

This module has been refactored to focus on the Multi-Effect Evaporator (MEE, unit 'E401')
for techno-economic and environmental analysis.
"""

import numpy as np
import pandas as pd
from chaospy import distributions as shape
import biosteam as bst
from biosteam.evaluation import Model, Metric
from biorefineries.VFA._process_settings import price, GWP_CFs, FEC_factors, load_preferences_and_process_settings
from biorefineries.VFA._tea_mee import create_vfa_tea_mee
from biorefineries.VFA._chemicals import chems
from biorefineries.VFA._systems_VFA import VFA_sys, F

#%% ---------------------------------------------------------------------------
# System and TEA Initialization
#----------------------------------------------------------------------------
load_preferences_and_process_settings()
sys = VFA_sys
tea = create_vfa_tea_mee(sys)
sys.operating_hours = tea.operating_days * 24

#%% ---------------------------------------------------------------------------
# Utility: Collect all streams
#----------------------------------------------------------------------------
all_streams = set(sys.feeds + sys.products)
for unit in sys.units:
    all_streams.update(unit.ins + unit.outs)
all_streams = list(all_streams)

# Set GWP CF on streams
for stream in all_streams:
    if stream.ID in GWP_CFs:
        stream.set_CF('GWP100', GWP_CFs[stream.ID], basis='kg', units='kg CO2e')

#%% ---------------------------------------------------------------------------
# Parametric Sweep: Number of Effects in MEE
#----------------------------------------------------------------------------
effects_options = [len(P) for P in [
    (101325, 73581),
    (101325, 73581, 50892),
    (101325, 73581, 50892, 32777)
]]
sweep_results = []
for n_effects in effects_options:
    # Update MEE pressure stages
    F.unit['E401'].P = F.unit['E401'].P[:n_effects]
    sys.reset_cache(); sys.empty_recycles(); sys.simulate()
    
    capex = tea.CAPEX
    opex = tea.OPEX
    mpsp = tea.solve_price(F.stored_vfa)
    gwp = calculate_total_GWP(all_streams, sys, F, GWP_CFs)
    fec = calculate_total_FEC(all_streams, F, FEC_factors)

    sweep_results.append({
        'Effects': n_effects,
        'CAPEX (USD)': capex,
        'OPEX (USD/yr)': opex,
        'MPSP (USD/kg)': mpsp,
        'GWP (kg CO2-eq/hr)': gwp,
        'FEC (kg oil-eq/hr)': fec
    })

sweep_df = pd.DataFrame(sweep_results)

#%% ---------------------------------------------------------------------------
# Environmental Impact Calculations for MEE
#----------------------------------------------------------------------------
def calculate_stream_GWP(stream_list, indicator='GWP100'):
    total = 0.0
    for s in stream_list:
        cf = s.get_CF(indicator) or 0.0
        total += s.F_mass * cf
    return total

def calculate_equipment_GWP(all_streams, sys, F, GWP_CFs):
    mee = F.unit['E401']
    breakdown = {}
    # Steam utilities
    if hasattr(mee, 'heat_utilities'):
        steam_cost = sum(u.cost for u in mee.heat_utilities.values()) * sys.operating_hours
        breakdown['Steam'] = steam_cost
    # Vacuum pump electricity
    if hasattr(mee, 'vacuum_system') and hasattr(mee.vacuum_system, 'power'):
        breakdown['Electricity'] = mee.vacuum_system.power * sys.operating_hours
    return sum(breakdown.values())

def calculate_total_GWP(streams, sys, F, GWP_CFs):
    return calculate_stream_GWP(streams) + calculate_equipment_GWP(streams, sys, F, GWP_CFs)

# FEC similarly for MEE

def calculate_stream_FEC(stream_list, indicator='FEC'):
    total = 0.0
    for s in stream_list:
        cf = s.get_CF(indicator) or 0.0
        total += s.F_mass * cf
    return total

# MEE equipment FEC: convert energy to oil-eq

def calculate_equipment_FEC(streams, F, FEC_factors):
    mee = F.unit['E401']
    breakdown = {}
    if hasattr(mee, 'heat_utilities'):
        steam = sum(u.flow * u.delta_H for u in mee.heat_utilities.values())
        breakdown['Steam'] = steam * FEC_factors.get('electricity', 0.0)
    if hasattr(mee, 'vacuum_system') and hasattr(mee.vacuum_system, 'power'):
        breakdown['Electricity'] = mee.vacuum_system.power * FEC_factors.get('electricity', 0.0)
    return sum(breakdown.values())

def calculate_total_FEC(streams, F, FEC_factors):
    return calculate_stream_FEC(streams) + calculate_equipment_FEC(streams, F, FEC_factors)

#%% ---------------------------------------------------------------------------
# Create Model and Sensitivity Analysis
#----------------------------------------------------------------------------
def create_model():
    metrics = [
        Metric('VFA Yield', lambda: F.stored_vfa.F_mass / F.feedstock.F_mass, 'kg/kg'),
        Metric('CAPEX', lambda: tea.CAPEX/1e6, 'Million USD'),
        Metric('OPEX', lambda: tea.OPEX/1e6, 'Million USD/yr'),
        Metric('MPSP', lambda: tea.solve_price(F.stored_vfa), 'USD/kg'),
        Metric('MEE GWP', lambda: calculate_total_GWP(all_streams, sys, F, GWP_CFs), 'kg CO2-eq/hr'),
        Metric('MEE FEC', lambda: calculate_total_FEC(all_streams, F, FEC_factors), 'kg oil-eq/hr')
    ]
    model = Model(sys, metrics)

    @model.parameter(name='MEE Effects', element=F.unit['E401'], kind='coupled', units='-',
                     baseline=len(F.unit['E401'].P), distribution=shape.DiscreteUniform(min(effects_options), max(effects_options)))
    def set_effects(n):
        F.unit['E401'].P = F.unit['E401'].P[:int(n)]

    @model.parameter(name='MEE Target Concentration', element=F.unit['E401'], kind='coupled', units='g/L',
                     baseline=F.unit['E401'].target_concentration,
                     distribution=shape.Triangle(0.5*F.unit['E401'].target_concentration,
                                                F.unit['E401'].target_concentration,
                                                1.5*F.unit['E401'].target_concentration))
    def set_target_conc(x):
        F.unit['E401'].target_concentration = x

    return model

model = create_model()

#%% ---------------------------------------------------------------------------
# Run and Save Results
#----------------------------------------------------------------------------
def model_specification():
    sys.reset_cache(); sys.empty_recycles(); sys.simulate()

def run_model(N=500, rule='L', notify_runs=10):
    np.random.seed(0)
    samples = model.sample(N, rule)
    model.load_samples(samples)
    model._specification = model_specification
    model.evaluate(notify=notify_runs)
    # Save tables
    model.table.to_excel('vfa_mee_model_results.xlsx')
    rho, p = model.spearman_r()
    rho.to_excel('vfa_mee_spearman_rho.xlsx')
    p.to_excel('vfa_mee_spearman_p.xlsx')

if __name__ == '__main__':
    run_model()
    sweep_df.to_excel('mee_parametric_sweep.xlsx', index=False)
}"}]}
