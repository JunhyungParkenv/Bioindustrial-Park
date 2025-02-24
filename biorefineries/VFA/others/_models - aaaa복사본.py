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
from biorefineries.VFA._process_settings import price, GWP_CFs, load_preferences_and_process_settings
from biorefineries.VFA._tea import create_vfa_tea
from biorefineries.VFA._chemicals import chems
from biorefineries.VFA._systems_VFA import VFA_sys, F
#%%
# =============================================================================
# System and TEA Initialization
# =============================================================================
load_preferences_and_process_settings()  # Flow 단위를 'kg/hr'로 설정
# 시스템 및 TEA 초기화 (시스템 모듈에서 고정 파라미터들도 이미 설정됨)
sys = VFA_sys
tea = create_vfa_tea(sys)
sys.operating_hours = tea.operating_days * 24

# 모든 스트림을 모읍니다.
all_streams = list(set(sys.feeds + sys.products))
for unit in sys.units:
    all_streams.extend(unit.ins + unit.outs)
# 중복 제거
all_streams = list(set(all_streams))

# 모든 스트림에 대해 GWP CF 설정
for stream in all_streams:
    if stream.ID in GWP_CFs:
        stream.set_CF('GWP100', GWP_CFs[stream.ID], basis='kg', units='kg CO2e')


#%%
# =============================================================================
# Create Model and Add Sensitivity & Fixed Parameters
# =============================================================================
def create_model():
    # Metric 정의 (민감도 분석 결과에 포함할 최종 지표들)
    metrics = [
        Metric('VFA Yield', lambda: F.stored_vfa.F_mass / F.feedstock.F_mass, 'kg/kg'),
        Metric('Electricity Consumption', lambda: sys.get_electricity_consumption(), 'kWh/yr'),
        Metric('Capital Investment (CAPEX)', lambda: tea.CAPEX / 1e6, 'Million USD'),
        Metric('Operating Cost (OPEX)', lambda: tea.OPEX / 1e6, 'Million USD/yr'),
        # Metric('Net Production Cost', lambda: tea.solve_price(F.stored_vfa), 'USD/kg'),
        Metric('GWP (Total)', 
               lambda: sys.get_total_feeds_impact('GWP100') * 1e3 / sys.operating_hours, 
               'g CO2-eq/hr'),
        Metric('MPSP (Minimum Product Selling Price)', lambda: tea.solve_price(F.stored_vfa), 'USD/kg') # Unit Conversion
    ]
    
    # 모델 생성 (민감도 분석 대상 파라미터 및 Metric 포함)
    model = Model(sys, metrics, exception_hook='raise')
    
    # 민감도 분석 파라미터 (이 변수들의 변화가 최종 결과에 미치는 영향을 평가)
    @model.parameter(name='ED Membrane Area',
                     element=F.S401,
                     kind='coupled',
                     units='m^2',
                     baseline=F.S401.A_m,
                     distribution=shape.Triangle(0.8 * F.S401.A_m, F.S401.A_m, 1.2 * F.S401.A_m))
    def set_ED_A_m(x):
        F.S401.A_m = x

    @model.parameter(name='ED Current Density',
                     element=F.S401,
                     kind='coupled',
                     units='A/m^2',
                     baseline=F.S401.j,
                     distribution=shape.Triangle(0.8 * F.S401.j, F.S401.j, 1.2 * F.S401.j))
    def set_ED_j(x):
        F.S401.j = x

    @model.parameter(name='AC Tank Residence Time',
                     element=F.T302,  # T302가 실제 AC 탱크 객체입니다.
                     kind='coupled',
                     units='hr',
                     baseline=F.T302.tau,
                     distribution=shape.Triangle(0.8 * F.T302.tau, F.T302.tau, 1.2 * F.T302.tau))
    def set_AC_tank_tau(x):
        F.T302.tau = x

    return model

model = create_model()

# =============================================================================
# Model Specification and Simulation
# =============================================================================
def model_specification():
    try:
        sys.simulate()
    except Exception as e:
        print("Error during simulation:", e)
        sys.reset_cache()
        sys.empty_recycles()
        sys.simulate()

def run_model(N=1000, rule='L', notify_runs=10, model=model):
    np.random.seed(1234)
    samples = model.sample(N, rule)
    model.load_samples(samples)
    model._specification = model_specification
    model.evaluate(notify=notify_runs)
    # Save evaluation results and Spearman correlation results to Excel files
    model.table.to_excel('vfa_model_results.xlsx')
    spearman_rho, p_values = model.spearman_r()
    spearman_rho.to_excel('vfa_spearman_rho.xlsx')
    p_values.to_excel('vfa_spearman_p.xlsx')
    return model

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == '__main__':
    run_model()