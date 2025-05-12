# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:15:09 2025

@author: Junhyung Park

This module has been refactored to focus on the Multi-Effect Evaporator (MEE, unit 'E401')
for techno-economic analysis (CAPEX/OPEX/MPSP).
"""

import numpy as np
import pandas as pd
from chaospy import distributions as shape
import biosteam as bst
from biosteam.evaluation import Model, Metric
from biorefineries.VFA._process_settings import load_preferences_and_process_settings

# 절대 임포트로 변경
from biorefineries.VFA._tea_MEE import create_vfa_tea_mee  
from biorefineries.VFA._systems_MEE import MEE_sys, F
#%% ---------------------------------------------------------------------------
# System and TEA Initialization
#----------------------------------------------------------------------------
load_preferences_and_process_settings()
sys = MEE_sys
tea = create_vfa_tea_mee(sys)
sys.operating_hours = tea.operating_days * 24
#%% ---------------------------------------------------------------------------
# 5년 수명 기준으로 시스템 전체 CAPEX를 모아서 연간 replacement 비용 계산
def calculate_system_replacement_opex(lifetime_years=5):
    total_capex = 0.0
    for unit in sys.units:
        if hasattr(unit, 'baseline_purchase_costs'):
            total_capex += sum(unit.baseline_purchase_costs.values())
    return total_capex / lifetime_years
#%%
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
    
    # 기본 OPEX + 5년 수명 replacement
    base_opex = tea.OPEX
    repl_opex = calculate_system_replacement_opex(lifetime_years=10)
    total_opex = base_opex + repl_opex
    
    sweep_results.append({
        'Effects': n_effects,
        'CAPEX (Million USD)': tea.CAPEX   / 1e6,
        'OPEX (Million USD/yr)': total_opex / 1e6,
        'MPSP (USD/kg)': tea.solve_price(F.stored_vfa)
    })

sweep_df = pd.DataFrame(sweep_results)

#%% ---------------------------------------------------------------------------
# Create Model and Sensitivity Analysis
#----------------------------------------------------------------------------
def create_model():
    metrics = [
        Metric('VFA Yield', lambda: F.stored_vfa.F_mass / F.feedstock.F_mass, 'kg/kg'),
        Metric('CAPEX', lambda: tea.CAPEX/1e6, 'Million USD'),
        Metric(
            'OPEX', 
            # base OPEX + replacement OPEX, 단위는 Million USD/yr
            lambda: (tea.OPEX + calculate_system_replacement_opex(10)) / 1e6, 
            'Million USD/yr'
        ),
        Metric('MPSP', lambda: tea.solve_price(F.stored_vfa), 'USD/kg')
    ]
    model = Model(sys, metrics)

    @model.parameter(
        name='MEE Effects', element=F.unit['E401'], kind='coupled', units='-',
        baseline=len(F.unit['E401'].P),
        distribution=shape.DiscreteUniform(min(effects_options), max(effects_options))
    )
    def set_effects(n):
        F.unit['E401'].P = F.unit['E401'].P[:int(n)]

    @model.parameter(
        name='MEE Target Concentration', element=F.unit['E401'], kind='coupled', units='g/L',
        baseline=F.unit['E401'].target_concentration,
        distribution=shape.Triangle(
            0.5*F.unit['E401'].target_concentration,
            F.unit['E401'].target_concentration,
            1.5*F.unit['E401'].target_concentration
        )
    )
    def set_target_conc(x):
        F.unit['E401'].target_concentration = x

    return model

model = create_model()

#%% ---------------------------------------------------------------------------
# Run and Save Results
#------------------------------------------------------------------------------
def model_specification():
    sys.reset_cache(); sys.empty_recycles(); sys.simulate()

def run_model(N=500, rule='L', notify_runs=10):
    np.random.seed(0)
    samples = model.sample(N, rule)
    model.load_samples(samples)
    model._specification = model_specification
    model.evaluate(notify=notify_runs)
    # 결과 저장
    model.table.to_excel('vfa_mee_model_results.xlsx')
    rho, p = model.spearman_r()
    rho.to_excel('vfa_mee_spearman_rho.xlsx')
    p.to_excel('vfa_mee_spearman_p.xlsx')

if __name__ == '__main__':
    # 1) 민감도 분석 & 스윕
    run_model()
    sweep_df.to_excel('mee_parametric_sweep.xlsx', index=False)
    print("▶ 스윕 결과 저장: mee_parametric_sweep.xlsx")

    # 2) 시스템 내 모든 유닛별 breakdown (Million USD 단위, 5년 수명 replacement 포함)
    lifetime_years = 10  
    with pd.ExcelWriter('mee_breakdown.xlsx') as w:
        for unit in sys.units:
            if not hasattr(unit, 'baseline_purchase_costs'):
                continue

            uid = unit.ID

            # CAPEX breakdown
            capex_musd = [(c, cost/1e6) 
                          for c, cost in unit.baseline_purchase_costs.items()]
            df_c = pd.DataFrame(capex_musd, columns=['Component','Cost (Million USD)'])

            # OPEX breakdown
            elec = unit.power_utility.cost * sys.operating_hours \
                   if hasattr(unit, 'power_utility') else 0.0
            maint = sum(unit.baseline_purchase_costs.values())*0.01
            repl  = sum(unit.baseline_purchase_costs.values())/lifetime_years
            opex_musd = [
                ('Electricity', elec/1e6),
                ('Maintenance', maint/1e6),
                ('Replacement', repl/1e6)
            ]
            df_o = pd.DataFrame(opex_musd, columns=['Component','Cost (Million USD/yr)'])

            df_c.to_excel(w, sheet_name=f'{uid}_CAPEX', index=False)
            df_o.to_excel(w, sheet_name=f'{uid}_OPEX', index=False)

    print("▶ 전체 유닛 breakdown 저장: mee_breakdown.xlsx")