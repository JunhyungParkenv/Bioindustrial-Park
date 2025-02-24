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
from biorefineries.VFA._process_settings import price, GWP_CFs, FEC_factors, load_preferences_and_process_settings
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
# 내부 함수: 모든 스트림의 GWP를 계산
def calculate_stream_GWP(stream_list, indicator='GWP100'):
    total = 0.0
    for stream in stream_list:
        if hasattr(stream, 'F_mass'):
            cf = stream.get_CF(indicator) or 0.0
            total += stream.F_mass * cf
    return total

# 내부 함수: 멤브레인 등 장비의 내재 GWP 계산 (CEM과 NF 모두 포함)
def calculate_equipment_GWP(F, GWP_CFs, lifetime_years=10):
    total_equipment_GWP = 0.0
    # CEM와 NF의 면적을 각각 확인합니다.
    # 만약 F.S401에 A_m_CEM, A_m_NF가 없다면, 전체 면적 A_m의 50%씩 사용하도록 합니다.
    for key in ['CEM', 'NF']:
        area_attr = f"A_m_{key}"
        if hasattr(F.S401, area_attr):
            area = getattr(F.S401, area_attr)
        else:
            area = F.S401.A_m * 0.5  # 기본값: 전체 면적의 50%
        cf = GWP_CFs.get(key, 0.0)  # kg CO2-eq/m²
        total_equipment_GWP += area * cf
    lifetime_hours = lifetime_years * 365 * 24
    return total_equipment_GWP / lifetime_hours  # kg CO2-eq/hr

# 최종 전체 GWP를 계산하는 함수 (스트림 + 장비)
def calculate_total_GWP(all_streams, sys, F, GWP_CFs, membrane_lifetime_years=10):
    stream_GWP = calculate_stream_GWP(all_streams, 'GWP100')
    equipment_GWP = calculate_equipment_GWP(F, GWP_CFs, lifetime_years=membrane_lifetime_years)
    total = stream_GWP + equipment_GWP
    # 최종 단위를 g CO2-eq/hr로 변환 및 시스템 운영 시간으로 나누기
    return total * 1e3 / sys.operating_hours
#%%
# --- FEC 계산 함수들 ---

# 전기 FEC: 시스템의 연간 전기 사용량(kWh/yr)을 MJ로 변환하고 FEC factor를 곱함
def calculate_electricity_FEC(sys, FEC_factors):
    # 1 kWh = 3.6 MJ
    consumption_MJ_per_year = sys.get_electricity_consumption() * 3.6
    # FEC in kg oil eq per year
    return consumption_MJ_per_year * FEC_factors['electricity']

# 스트림 기반 FEC 계산 (필요 시; 현재는 주로 전기 및 장비 FEC를 고려)
def calculate_stream_FEC(stream_list, indicator='FEC'):
    total = 0.0
    for stream in stream_list:
        if hasattr(stream, 'F_mass'):
            cf = stream.get_CF(indicator) or 0.0
            total += stream.F_mass * cf
    return total

# 장비(멤브레인 등) FEC 계산  
def calculate_equipment_FEC(F, FEC_factors, lifetime_years=10, 
                            membrane_mass_factor=5.0,  # kg membrane per m² (가정)
                            electrode_fraction=0.1,    # 전체 membrane mass의 10%
                            current_collector_fraction=0.05):  # 전체 membrane mass의 5%
    # F.S401는 ED 유닛 (멤브레인 unit)으로 가정
    # A_m_CEM, A_m_NF가 없다면 전체 면적 A_m의 50%씩 할당
    if hasattr(F.S401, 'A_m_CEM'):
        A_m_CEM = F.S401.A_m_CEM
    else:
        A_m_CEM = F.S401.A_m * 0.5
    if hasattr(F.S401, 'A_m_NF'):
        A_m_NF = F.S401.A_m_NF
    else:
        A_m_NF = F.S401.A_m * 0.5
    # membrane mass (kg) = area (m²) * membrane_mass_factor (kg/m²)
    mass_membrane = (A_m_CEM + A_m_NF) * membrane_mass_factor
    # Electrode and Current Collector mass assumptions:
    mass_electrode = mass_membrane * electrode_fraction
    mass_cc = mass_membrane * current_collector_fraction
    # FEC contributions (kg oil eq per year)
    lifetime_hours = lifetime_years * 365 * 24
    FEC_membrane = mass_membrane * FEC_factors['Membrane'] / lifetime_hours
    FEC_electrode = mass_electrode * FEC_factors['Electrode'] / lifetime_hours
    FEC_cc = mass_cc * FEC_factors['Current Collector'] / lifetime_hours
    return FEC_membrane + FEC_electrode + FEC_cc

# 최종 전체 FEC 계산 함수: 전기 FEC + (스트림 FEC, 필요 시) + 장비 FEC  
def calculate_total_FEC(sys, all_streams, F, FEC_factors, lifetime_years=10, 
                        membrane_mass_factor=5.0, electrode_fraction=0.1, current_collector_fraction=0.05):
    elec_FEC = calculate_electricity_FEC(sys, FEC_factors)
    # stream_FEC = calculate_stream_FEC(all_streams, 'FEC')  # 사용하려면, 해당 CF가 스트림에 설정되어야 함
    equip_FEC = calculate_equipment_FEC(F, FEC_factors, lifetime_years, membrane_mass_factor, electrode_fraction, current_collector_fraction)
    total_FEC_yearly = elec_FEC + equip_FEC
    # 시간당 FEC (kg oil eq/hr)
    return total_FEC_yearly / sys.operating_hours

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
               lambda: calculate_total_GWP(all_streams, sys, F, GWP_CFs, membrane_lifetime_years=10),
               'g CO2-eq/hr'),
        # 새로 추가: FEC Metric (총 Fossil Energy Consumption)
        Metric('FEC (Total)', 
               lambda: calculate_total_FEC(sys, all_streams, F, FEC_factors, lifetime_years=10),
               'kg oil eq/hr'),
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