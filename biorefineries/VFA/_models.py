#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:15:09 2025

@author: Junhyung Park
"""

import numpy as np
import pandas as pd
from chaospy import distributions as shape
import biosteam as bst
from biosteam.evaluation import Model, Metric
from biosteam.evaluation.evaluation_tools.parameter import Setter
from biorefineries.VFA._process_settings import price, GWP_CFs, load_preferences_and_process_settings
from biorefineries.VFA._tea import create_vfa_tea
from biorefineries.VFA._chemicals import chems
from biorefineries.VFA._systems_VFA import VFA_sys, F

# =============================================================================
# System and TEA Initialization
# =============================================================================

# Load process preferences and settings
load_preferences_and_process_settings()  # Flow 단위를 'kg/hr'로 설정

# Initialize VFA system and TEA
# 시스템 및 TEA 초기화
sys = VFA_sys
tea = create_vfa_tea(sys)
sys.operating_hours = tea.operating_days * 24

# =============================================================================
# Metrics and Evaluation
# =============================================================================

# Define metrics
metrics = [
    Metric('VFA Yield', lambda: F.stored_vfa.F_mass / F.feedstock.F_mass, 'kg/kg'),
    Metric('Electricity Consumption', lambda: sys.get_electricity_consumption(), 'kWh/yr'),
    Metric('Capital Investment (CAPEX)', lambda: tea.CAPEX / 1e6, 'Million USD'),
    Metric('Operating Cost (OPEX)', lambda: tea.OPEX / 1e6, 'Million USD/yr'),
    Metric('Net Production Cost', lambda: tea.solve_price(F.stored_vfa), 'USD/kg'),
    Metric('Global Warming Potential', lambda: sys.get_total_feeds_impact('GWP100') * 1e3 / sys.operating_hours, 'g CO2-eq/hr'),
    Metric('MPSP (Minimum Product Selling Price)', lambda: tea.solve_price(F.stored_vfa), 'USD/kg'),  # MPSP 추가
]

# =============================================================================
# Parameter Definitions
# =============================================================================

def create_model():
    """
    Create a BioSTEAM model for the VFA system.
    Includes system parameters, TEA inputs, and sampling distributions.
    """
    model = Model(sys, metrics, exception_hook='raise')
    param = model.parameter

    # Feedstock Flow
    D = shape.Triangle(10000, 12000, 14000)  # Example range (kg/hr)
    @param(name='Feedstock Flow', element=F.feedstock, kind='coupled', units='kg/hr',
           baseline=12000, distribution=D)
    def set_feedstock_flow(flow):
        F.feedstock.F_mass = flow

    # Electricity Price
    D = shape.Triangle(0.06, 0.07, 0.08)  # Example range ($/kWh)
    @param(name='Electricity Price', element='Electricity', kind='isolated', units='$/kWh',
           baseline=0.07, distribution=D)
    def set_electricity_price(price):
        bst.PowerUtility.price = price

    # VFA Selling Price
    D = shape.Triangle(2.0, 2.5, 3.0)  # Example range ($/kg)
    @param(name='VFA Selling Price', element=F.stored_vfa, kind='isolated', units='$/kg',
           baseline=2.5, distribution=D)
    def set_vfa_price(price):
        F.stored_vfa.price = price

    # Operating Days
    D = shape.Uniform(320, 350)  # Example range (days/yr)
    @param(name='Operating Days', element='TEA', kind='isolated', units='days/yr',
           baseline=350, distribution=D)
    def set_operating_days(days):
        tea.operating_days = days
        sys.operating_hours = days * 24

    return model

# Create the model
model = create_model()

# =============================================================================
# Model Specification and Simulation
# =============================================================================

def model_specification():
    """
    Custom specification function to handle errors during simulation.
    """
    try:
        sys.simulate()
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        sys.reset_cache()
        sys.empty_recycles()
        sys.simulate()

def run_model(N=1000, rule='L', notify_runs=10, model=model):
    """
    Run the model with Monte Carlo or Latin Hypercube sampling.
    """
    np.random.seed(1234)  # For reproducibility
    samples = model.sample(N, rule)
    model.load_samples(samples)
    model._specification = model_specification
    model.evaluate(notify=notify_runs)

    # Save results to Excel
    model.table.to_excel('vfa_model_results.xlsx')
    spearman_rho, p_values = model.spearman_r()
    spearman_rho.to_excel('vfa_spearman_rho.xlsx')
    p_values.to_excel('vfa_spearman_p.xlsx')

    return model

# =============================================================================
# ED CAPEX Breakdown Print Function
# =============================================================================
def save_ed_capex_breakdown_to_excel(filename='ed_capex_breakdown.xlsx'):
    """
    Save the ED unit's CAPEX breakdown (cost components) to an Excel file.
    """
    ed_breakdown = tea.ED_CAPEX_breakdown
    if ed_breakdown:
        df = pd.DataFrame.from_dict(ed_breakdown, orient='index', columns=['Cost (USD)'])
        df.index.name = 'Component'
        df.to_excel(filename)
        print(f"ED CAPEX breakdown saved to {filename}")
    else:
        print("No ED CAPEX breakdown available.")

        
# =============================================================================
# Main Execution
# =============================================================================
# Run the model
if __name__ == '__main__':
    run_model()
    # 추가: ED CAPEX breakdown을 Excel 파일로 저장
save_ed_capex_breakdown_to_excel()