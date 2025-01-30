# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 02:35:32 2025

@author: Junhyung Park
"""

# -*- coding: utf-8 -*-
"""
Electrodialysis (ED) Process Analysis Module

Includes:
- TEA analysis for ED unit
- CAPEX & OPEX breakdown
- Sensitivity analysis for membrane area, current density, flux
- Uncertainty analysis for economic parameters
"""

import numpy as np
import pandas as pd
import biosteam as bst
import thermosteam as tmo
from biosteam.evaluation import Model, Metric
from chaospy import distributions as shape
from biorefineries.VFA._units import ED  # 기존 ED 유닛 불러오기
from biorefineries.VFA._tea import VFA_TEA
from biorefineries.VFA._process_settings import price

# =============================================================================
# ED TEA Class (Extending from ED Unit)
# =============================================================================

class ED_TEA(ED):
    """
    ED Techno-Economic Analysis (TEA) model.
    Inherits from the existing ED unit and extends with cost breakdown.
    """
    def __init__(self, ID='', ins=None, outs=(), **kwargs):
        super().__init__(ID, ins, outs, **kwargs)
        self.tea = VFA_TEA(self)  # 기존 TEA 시스템 적용

    @property
    def CAPEX(self):
        """Use the existing CAPEX from the inherited ED unit."""
        return sum(self._cost)  # @cost 데코레이터가 정의한 CAPEX

    @property
    def OPEX(self):
        """Calculate annual operating costs (electricity, maintenance, labor)."""
        electricity_cost = self.design_results['Power consumption'] * price['electricity']
        maintenance_cost = 0.03 * self.CAPEX  # Maintenance: 3% of CAPEX
        labor_cost = 1e6  # Assumed fixed labor cost
        return electricity_cost + maintenance_cost + labor_cost

    def get_capex_breakdown(self):
        """Return detailed CAPEX breakdown."""
        return {
            'Membrane Cost': 100 * self.A_m,
            'Power Supply Cost': 20 * self.A_m,
            'Electrode Cost': 50 * self.A_m,
            'Frame Cost': 10 * self.A_m,
            'Installation Cost': 0.2 * (100 * self.A_m + 20 * self.A_m + 50 * self.A_m + 10 * self.A_m)
        }

# =============================================================================
# Sensitivity Analysis: Membrane Area vs Flux
# =============================================================================

def find_optimal_membrane_area(j=5.058, target_flux=0.01):
    """
    Optimize membrane area to achieve the target flux.
    """
    A_values = np.linspace(1, 100, 50)  # Membrane area from 1 to 100 m²
    flux_values = [j * A / (96485.3 * 1) for A in A_values]  # Simplified flux calculation

    optimal_A = A_values[np.argmin(np.abs(np.array(flux_values) - target_flux))]
    return optimal_A, A_values, flux_values

# =============================================================================
# Monte Carlo & Sensitivity Analysis
# =============================================================================

def create_ed_model():
    """
    Create a BioSTEAM model for the ED process.
    """
    ed_unit = ED_TEA('ED_Unit')
    metrics = [
        Metric('CAPEX', lambda: ed_unit.CAPEX / 1e6, 'Million USD'),
        Metric('OPEX', lambda: ed_unit.OPEX / 1e6, 'Million USD/yr'),
        Metric('Membrane Area', lambda: ed_unit.A_m, 'm²'),
        Metric('Flux', lambda: ed_unit.j * ed_unit.A_m / (96485.3 * ed_unit.z_T), 'mol/m²/s'),
        Metric('Current Density', lambda: ed_unit.j, 'A/m²'),
        Metric('Power Consumption', lambda: ed_unit.design_results['Power consumption'], 'W'),
    ]

    model = Model(ed_unit, metrics, exception_hook='raise')
    param = model.parameter

    # Membrane Area Distribution
    D = shape.Triangle(5, 10, 20)
    @param(name='Membrane Area', element=ed_unit, kind='coupled', units='m²',
           baseline=10, distribution=D)
    def set_membrane_area(A_m):
        ed_unit.A_m = A_m

    # Current Density Distribution
    D = shape.Uniform(3, 7)
    @param(name='Current Density', element=ed_unit, kind='coupled', units='A/m²',
           baseline=5.058, distribution=D)
    def set_current_density(j):
        ed_unit.j = j

    return model

# =============================================================================
# Run the Model
# =============================================================================

def run_ed_analysis(N=500):
    """
    Run sensitivity and uncertainty analysis for the ED process.
    """
    model = create_ed_model()
    np.random.seed(42)
    samples = model.sample(N, 'L')  # Latin Hypercube Sampling
    model.load_samples(samples)
    model.evaluate()

    # Save results
    model.table.to_excel('ed_analysis_results.xlsx')
    spearman_rho, p_values = model.spearman_r()
    spearman_rho.to_excel('ed_spearman_rho.xlsx')
    p_values.to_excel('ed_spearman_p.xlsx')

    return model

# =============================================================================
# Execution
# =============================================================================

if __name__ == '__main__':
    optimal_A, A_values, flux_values = find_optimal_membrane_area()
    print(f"Optimal Membrane Area: {optimal_A:.2f} m²")
    
    model = run_ed_analysis()
    print("ED Analysis Complete. Results saved to Excel.")
