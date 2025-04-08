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
        
#%% =============================================================================
# Parametric Sweep: Current Density vs. Membrane Area
# =============================================================================
# 전류 밀도를 10 A/m² ~ 500 A/m² 범위에서 100단계로 변화시키면서, 각 조건에서 시뮬레이션 실행 후 결과 기록
# baseline 값 저장 (반복문 전에 한 번 저장)
baseline_j = F.S401.j              # 기존의 baseline current density
baseline_tau = F.T302.tau          # 기존의 AC 탱크 HRT

current_density_values = np.linspace(10, 500, 100)
sweep_results = []

for cd in current_density_values:
    # 현재 전류 밀도 설정
    F.S401.j = cd
    # 시스템 초기화 후 시뮬레이션
    sys.reset_cache()
    sys.empty_recycles()
    sys.simulate()

    # 시뮬레이션 후 TEA 정보를 업데이트 (CAPEX, OPEX, MPSP 등)
    capex = tea.CAPEX             # USD
    opex = tea.OPEX              # USD/yr
    mpsp = tea.solve_price(F.stored_vfa)  # USD/kg
    
    # 최종 VFA의 농도 계산 (예: VFA 스트림의 질량/부피, 단위: kg/m³)
    # 단, F_vol 속성이 존재한다고 가정
    try:
        vfa_conc = F.stored_vfa.F_mass / F.stored_vfa.F_vol
    except Exception:
        vfa_conc = np.nan
    
    # 전류(I) 계산 및 개별 이온 플럭스 계산 (I = 전류 밀도 * 멤브레인 면적)
    I = F.S401.j * F.S401.A_m
    flux_dict = F.S401.calculate_flux(I)
    
    # 결과를 한 행(row)에 기록
    row = {
        "Current Density (A/m²)": cd,
        "Membrane Area (m²)": F.S401.A_m,
        "AC Tank HRT (hr)": F.T302.tau,
        "Final VFA Concentration (kg/m³)": vfa_conc,
        "CAPEX (USD)": capex,
        "OPEX (USD/yr)": opex,
        "MPSP (USD/kg)": mpsp
    }
    
    # 각 이온 플럭스 값도 추가 (단위: mol/m²/s)
    for ion, flux in flux_dict.items():
        row[f"{ion} Flux (mol/m²/s)"] = flux
        
    sweep_results.append(row)
    print(f"Current Density = {cd:.2f} A/m² → Membrane Area = {F.S401.A_m:.4f} m², AC Tank HRT = {F.T302.tau:.4f} hr")
        
sweep_df = pd.DataFrame(sweep_results) 
#%%
# =============================================================================
# ED GWP Calculation (절대값: kg CO2-eq)
# =============================================================================
def calculate_stream_GWP(stream_list, indicator='GWP100'):
    total = 0.0
    for stream in stream_list:
        if hasattr(stream, 'F_mass'):
            cf = stream.get_CF(indicator) or 0.0
            # kg/hr * kg CO2-eq/kg → kg CO2-eq/hr
            total += stream.F_mass * cf
    # 연간 곱셈(sys.operating_hours) 제거 → 절대값 반환
    return total

def calculate_equipment_GWP(F, GWP_CFs, lifetime_years=5):
    if hasattr(F.S401, 'A_m_CEM'):
        A_m_CEM = F.S401.A_m_CEM
    else:
        A_m_CEM = F.S401.A_m * 0.5
    if hasattr(F.S401, 'A_m_NF'):
        A_m_NF = F.S401.A_m_NF
    else:
        A_m_NF = F.S401.A_m * 0.5
    total_membrane_area = A_m_CEM + A_m_NF  # m²
    lifetime_hours = lifetime_years * 365 * 24
    # 각 항목의 시간당 기여 (kg CO2-eq/hr)
    hourly_CEM = A_m_CEM * GWP_CFs.get('CEM', 0.0) / lifetime_hours
    hourly_NF  = A_m_NF  * GWP_CFs.get('NF', 0.0)  / lifetime_hours
    base_area = F.S401.A_m if hasattr(F.S401, 'A_m') else total_membrane_area
    hourly_electrode         = base_area * GWP_CFs.get('Electrode', 0.0)         / (lifetime_hours * 2)
    hourly_current_collector = base_area * GWP_CFs.get('Current Collector', 0.0) / (lifetime_hours * 2)
    hourly_FeCN              = base_area * GWP_CFs.get('Fe(CN)', 0.0)              / (lifetime_hours * 4)
    hourly_frames            = base_area * GWP_CFs.get('Frames', 0.0)            / (lifetime_hours * 4)
    hourly_total = hourly_CEM + hourly_NF + hourly_electrode + hourly_current_collector + hourly_FeCN + hourly_frames
    # 연간 곱셈 제거 → 시간당 절대값 반환
    return hourly_total

def calculate_ED_electricity_GWP(F, GWP_CFs, sys):
    # F.S401.power_utility.consumption: kW = kWh/hr
    ed_consumption = F.S401.power_utility.consumption
    # 연간 곱셈 제거: kWh/hr * CF → kg CO2-eq/hr
    total_GWP = ed_consumption * GWP_CFs['electricity']
    return total_GWP

def calculate_total_GWP(all_streams, sys, F, GWP_CFs, membrane_lifetime_years=10):
    stream_GWP = calculate_stream_GWP(all_streams, 'GWP100')
    equipment_GWP = calculate_equipment_GWP(F, GWP_CFs, lifetime_years=membrane_lifetime_years)
    electricity_GWP = calculate_ED_electricity_GWP(F, GWP_CFs, sys)
    total = stream_GWP + equipment_GWP + electricity_GWP
    # AC & DC 탱크 stainless steel 기여 (절대값: kg CO2-eq)
    ac_tank = next((u for u in sys.units if u.ID == 'T302'), None)
    dc_tank = next((u for u in sys.units if u.ID == 'T301'), None)
    if ac_tank is not None and dc_tank is not None:
        ac_volume = ac_tank.design_results.get('Volume', 0)
        dc_volume = dc_tank.design_results.get('Volume', 0)
        total_volume = ac_volume + dc_volume
        stainless_mass = total_volume * 500  # kg (500 kg/m³)
        lifetime_hours_tanks = 20 * 365 * 24
        stainless_mass_per_hr = stainless_mass / lifetime_hours_tanks  # kg/hr
        stainless_gwp = stainless_mass_per_hr * GWP_CFs.get('StainlessSteel', 6.0)
        total += stainless_gwp
    return total  # 단위: kg CO2-eq (절대값, 1시간 기준)

# ED + AC/DC Tanks Breakdown
def get_ED_GWP_breakdown(F, sys, GWP_CFs, lifetime_years=5):
    if hasattr(F.S401, 'A_m_CEM'):
        A_m_CEM = F.S401.A_m_CEM
    else:
        A_m_CEM = F.S401.A_m * 0.5
    if hasattr(F.S401, 'A_m_NF'):
        A_m_NF = F.S401.A_m_NF
    else:
        A_m_NF = F.S401.A_m * 0.5
    lifetime_hours = lifetime_years * 365 * 24
    breakdown = {}
    # 각 항목의 시간당 값을 그대로 반환 (절대값)
    breakdown['CEM'] = A_m_CEM * GWP_CFs.get('CEM', 0.0) / lifetime_hours
    breakdown['NF'] = A_m_NF * GWP_CFs.get('NF', 0.0) / lifetime_hours
    base_area = F.S401.A_m if hasattr(F.S401, 'A_m') else (A_m_CEM + A_m_NF)
    breakdown['Electrode'] = base_area * GWP_CFs.get('Electrode', 0.0) / (lifetime_hours * 2)
    breakdown['Current Collector'] = base_area * GWP_CFs.get('Current Collector', 0.0) / (lifetime_hours * 2)
    breakdown['Fe(CN)'] = base_area * GWP_CFs.get('Fe(CN)', 0.0) / (lifetime_hours * 4)
    breakdown['Frames'] = base_area * GWP_CFs.get('Frames', 0.0) / (lifetime_hours * 4)
    ac_tank = next((u for u in sys.units if u.ID == 'T302'), None)
    dc_tank = next((u for u in sys.units if u.ID == 'T301'), None)
    if ac_tank is not None and dc_tank is not None:
        ac_volume = ac_tank.design_results.get('Volume', 0)
        dc_volume = dc_tank.design_results.get('Volume', 0)
        total_volume = ac_volume + dc_volume
        stainless_mass = total_volume * 8  # m3 * 8 kg/m3 = kg (mass)
        lifetime_hours_tanks = 20 * 365 * 24
        stainless_mass_per_hr = stainless_mass / lifetime_hours_tanks
        breakdown['StainlessSteel'] = stainless_mass_per_hr * GWP_CFs.get('StainlessSteel', 0.0)
    return breakdown

#%%
# =============================================================================
# ED FEC Calculation (절대값: kg oil-eq)
# =============================================================================
def calculate_stream_FEC(stream_list, indicator='FEC'):
    total = 0.0
    for stream in stream_list:
        if hasattr(stream, 'F_mass'):
            cf = stream.get_CF(indicator) or 0.0
            total += stream.F_mass * cf
    return total

def calculate_equipment_FEC(F, FEC_factors, lifetime_years=5, 
                            membrane_mass_factor=5.0, electrode_fraction=0.1, current_collector_fraction=0.05):
    if hasattr(F.S401, 'A_m_CEM'):
        A_m_CEM = F.S401.A_m_CEM
    else:
        A_m_CEM = F.S401.A_m * 0.5
    if hasattr(F.S401, 'A_m_NF'):
        A_m_NF = F.S401.A_m_NF
    else:
        A_m_NF = F.S401.A_m * 0.5
    mass_membrane_CEM = A_m_CEM * membrane_mass_factor
    mass_membrane_NF = A_m_NF * membrane_mass_factor
    total_membrane_mass = mass_membrane_CEM + mass_membrane_NF
    lifetime_hours = lifetime_years * 365 * 24
    FEC_membrane_CEM = mass_membrane_CEM * FEC_factors.get('CEM', 0.0) / lifetime_hours
    FEC_membrane_NF  = mass_membrane_NF  * FEC_factors.get('NF', 0.0)  / lifetime_hours
    FEC_electrode    = total_membrane_mass * electrode_fraction * FEC_factors.get('Electrode', 0.0) / (lifetime_hours * 2)
    FEC_cc           = total_membrane_mass * current_collector_fraction * FEC_factors.get('Current Collector', 0.0) / (lifetime_hours * 2)
    FEC_FeCN         = total_membrane_mass * 0.05 * FEC_factors.get('Fe(CN)', 0.0) / (lifetime_hours * 4)
    FEC_frames       = total_membrane_mass * 0.10 * FEC_factors.get('Frames', 0.0) / (lifetime_hours * 4)
    total_equipment_FEC = (FEC_membrane_CEM + FEC_membrane_NF +
                           FEC_electrode + FEC_cc +
                           FEC_FeCN + FEC_frames)
    return total_equipment_FEC

def calculate_ED_electricity_FEC(F, FEC_factors, sys):
    ed_consumption = F.S401.power_utility.consumption  # kW = kWh/hr
    consumption_MJ = ed_consumption * 3.6  # MJ/hr
    return consumption_MJ * FEC_factors['electricity']

def calculate_total_FEC(sys, all_streams, F, FEC_factors, lifetime_years=10, 
                        membrane_mass_factor=5.0, electrode_fraction=0.1, current_collector_fraction=0.05):
    elec_FEC = calculate_ED_electricity_FEC(F, FEC_factors, sys)
    equip_FEC = calculate_equipment_FEC(F, FEC_factors, lifetime_years, 
                                        membrane_mass_factor, electrode_fraction, current_collector_fraction)
    stream_FEC = calculate_stream_FEC(all_streams, 'FEC')
    total_FEC = elec_FEC + equip_FEC + stream_FEC
    try:
        ac_tank = next((u for u in sys.units if u.ID == 'T302'), None)
        dc_tank = next((u for u in sys.units if u.ID == 'T301'), None)
    except KeyError:
        ac_tank = dc_tank = None
    if ac_tank is not None and dc_tank is not None:
        ac_volume = ac_tank.design_results.get('Volume', 0)
        dc_volume = dc_tank.design_results.get('Volume', 0)
        total_volume = ac_volume + dc_volume
        stainless_mass = total_volume * 500  # kg
        lifetime_hours_tanks = 20 * 365 * 24
        stainless_mass_per_hr = stainless_mass / lifetime_hours_tanks  # kg/hr
        stainless_fec = stainless_mass_per_hr * FEC_factors.get('StainlessSteel', 0.3)
        total_FEC += stainless_fec
    return total_FEC  # 단위: kg oil-eq (절대값)

# ED + AC/DC Tanks Breakdown
def get_ED_FEC_breakdown(F, sys, FEC_factors, lifetime_years=5, 
                         membrane_mass_factor=5.0, electrode_fraction=0.1, current_collector_fraction=0.05):
    if hasattr(F.S401, 'A_m_CEM'):
        A_m_CEM = F.S401.A_m_CEM
    else:
        A_m_CEM = F.S401.A_m * 0.5
    if hasattr(F.S401, 'A_m_NF'):
        A_m_NF = F.S401.A_m_NF
    else:
        A_m_NF = F.S401.A_m * 0.5
    lifetime_hours = lifetime_years * 365 * 24
    breakdown = {}
    breakdown['CEM'] = A_m_CEM * FEC_factors.get('CEM', 0.0) / lifetime_hours
    breakdown['NF'] = A_m_NF * FEC_factors.get('NF', 0.0) / lifetime_hours
    base_area = F.S401.A_m if hasattr(F.S401, 'A_m') else (A_m_CEM + A_m_NF)
    breakdown['Electrode'] = base_area * FEC_factors.get('Electrode', 0.0) / (lifetime_hours * 2)
    breakdown['Current Collector'] = base_area * FEC_factors.get('Current Collector', 0.0) / (lifetime_hours * 2)
    breakdown['Fe(CN)'] = base_area * FEC_factors.get('Fe(CN)', 0.0) / (lifetime_hours * 4)
    breakdown['Frames'] = base_area * FEC_factors.get('Frames', 0.0) / (lifetime_hours * 4)
    ac_tank = next((u for u in sys.units if u.ID == 'T302'), None)
    dc_tank = next((u for u in sys.units if u.ID == 'T301'), None)
    if ac_tank is not None and dc_tank is not None:
        ac_volume = ac_tank.design_results.get('Volume', 0)
        dc_volume = dc_tank.design_results.get('Volume', 0)
        total_volume = ac_volume + dc_volume
        stainless_mass = total_volume * 8  # m3 * 8 kg/m3 = kg (mass)
        lifetime_hours_tanks = 20 * 365 * 24
        stainless_mass_per_hr = stainless_mass / lifetime_hours_tanks
        breakdown['StainlessSteel'] = stainless_mass_per_hr * FEC_factors.get('StainlessSteel', 0.0)
    return breakdown

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
        Metric('ED GWP', 
               lambda: calculate_total_GWP(all_streams, sys, F, GWP_CFs, membrane_lifetime_years=F.S401.lifetime),
               'kg CO2-eq'),
        Metric('ED FEC', 
               lambda: calculate_total_FEC(sys, all_streams, F, FEC_factors, lifetime_years=F.S401.lifetime),
               'kg oil-eq'),
        Metric('MPSP (Minimum Product Selling Price)', lambda: tea.solve_price(F.stored_vfa), 'USD/kg'),
        Metric('SEC', 
               lambda: sys.get_electricity_consumption() / (F.stored_vfa.F_mass * sys.operating_hours), 
               'kWh/kg')
    ]
    model = Model(sys, metrics, exception_hook='raise')
    
    # 민감도 분석 파라미터 (이 변수들의 변화가 최종 결과에 미치는 영향을 평가)
    # 민감도 파라미터: ED Membrane Lifetime (CEM의 수명, 기본 5년)
    @model.parameter(name='ED Membrane Lifetime',
                     element=F.S401,
                     kind='coupled',
                     units='yr',
                     baseline=5,
                     distribution=shape.Triangle(4, 5, 6))
    def set_ED_lifetime(x):
        F.S401.lifetime = x  # 이 값이 교체 비용 계산에 사용됨
        
    @model.parameter(name='ED Membrane Resistance (r_m)',
                     element=F.S401,
                     kind='coupled',
                     units='Ohm*m^2',
                     baseline=F.S401.r_m,
                     distribution=shape.Triangle(0.8 * F.S401.r_m, F.S401.r_m, 1.2 * F.S401.r_m))
    def set_ED_r_m(x):
        F.S401.r_m = x
        
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
        
    # @model.parameter(name='ED Current',
    #                  element=F.S401,
    #                  kind='coupled',
    #                  units='A',
    #                  baseline=F.S401.j * F.S401.A_m,
    #                  distribution=shape.Triangle(0.8 * F.S401.j * F.S401.A_m, 
    #                                              F.S401.j * F.S401.A_m, 
    #                                              1.2 * F.S401.j * F.S401.A_m))
    # def set_ED_I(x):
    #     F.S401.j = x / F.S401.A_m

    # @model.parameter(name='AC Tank Residence Time',
    #                  element=F.T302,  # T302가 실제 AC 탱크 객체입니다.
    #                  kind='coupled',
    #                  units='hr',
    #                  baseline=F.T302.tau,
    #                  distribution=shape.Triangle(0.8 * F.T302.tau, F.T302.tau, 1.2 * F.T302.tau))
    # def set_AC_tank_tau(x):
    #     F.T302.tau = x
    
    # ────── 민감도 파라미터: ED Current Efficiency (CE) ──────
    # F.S401 객체에 CE 속성이 없으면 기본값 0.8 (80% 효율)으로 설정
    baseline_CE = getattr(F.S401, 'CE', 0.8)
    
    @model.parameter(name='ED Current Efficiency',
                     element=F.S401,
                     kind='coupled',
                     units='-',
                     baseline=baseline_CE,
                     distribution=shape.Triangle(0.8 * baseline_CE, baseline_CE, 1.2 * baseline_CE))
    def set_ED_CE(x):
        F.S401.CE = x
        
    @model.parameter(name='AC Tank Residence Time',
                     element=F.T302,  # T302가 실제 AC 탱크 객체입니다.
                     kind='coupled',
                     units='hr',
                     baseline=F.T302.tau,
                     distribution=shape.Triangle(0.9 * (F.T302.tau if F.T302.tau > 1e-6 else 1e-6),
                                                  (F.T302.tau if F.T302.tau > 1e-6 else 1e-6),
                                                  1.2 * (F.T302.tau if F.T302.tau > 1e-6 else 1e-6)))
    def set_AC_tank_tau(x):
        F.T302.tau = x

    return model

model = create_model()
# =============================================================================
# Membrane (CEM+NF) Lifetime
# =============================================================================
# Monkey-Patch: ED_OPEX_breakdown 수정 (CEM 비용에 대해 lifetime 사용)
def new_ED_OPEX_breakdown(self):
    ed_unit = next((u for u in sys.units if u.ID == 'S401'), None)
    if ed_unit is None or not hasattr(ed_unit, 'baseline_purchase_costs'):
        return {}
    bp = ed_unit.baseline_purchase_costs
    lifetime = getattr(F.S401, 'lifetime', 5)  # 기본값 5년
    new_breakdown = {}
    if 'CEM' in bp:
        new_breakdown['CEM'] = bp['CEM'] / lifetime  # lifetime 적용
    if 'NF' in bp:
        new_breakdown['NF'] = bp['NF'] / lifetime  # 만약 NF도 동일하게 적용하고 싶다면
    if 'Coating Solution' in bp:
        new_breakdown['Coating Solution'] = bp['Coating Solution'] / lifetime
    if 'Electrode' in bp:
        new_breakdown['Electrode'] = bp['Electrode'] / (lifetime * 2)
    if 'Current Collector' in bp:
        new_breakdown['Current Collector'] = bp['Current Collector'] / (lifetime * 2)
    if 'Frames' in bp:
        new_breakdown['Frames'] = bp['Frames'] / (lifetime * 4)
    if 'Fe(CN)' in bp:
        new_breakdown['Fe(CN)'] = bp['Fe(CN)'] / (lifetime * 4)
    if ed_unit is not None and hasattr(ed_unit, 'power_utility'):
        ed_elec_cost = ed_unit.power_utility.cost * sys.operating_hours
        new_breakdown['Electricity'] = ed_elec_cost
    return new_breakdown

# ED_OPEX_breakdown 속성을 override
tea.__class__.ED_OPEX_breakdown = property(new_ED_OPEX_breakdown)

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
    
    # ED 구성 부품별 비용 출력 (단일 시뮬레이션 결과)
    # run_and_print_ED_breakdown()
    return model

#%% =============================================================================
# Main Execution
# =============================================================================
# 각 unit의 OPEX에 property insurance와 maintenance는 추가 가능 (_tea 모듈 참고)
if __name__ == '__main__':
    run_model()
    # 기존 ED, AC 관련 breakdown 정보 추출
    ed_capex = tea.ED_CAPEX_breakdown
    ed_opex = tea.ED_OPEX_breakdown
    ac_capex = tea.AC_CAPEX_breakdown
    ac_opex = tea.AC_OPEX_breakdown
    ed_gwp_breakdown = get_ED_GWP_breakdown(F, sys, GWP_CFs, lifetime_years=F.S401.lifetime)
    ed_fec_breakdown = get_ED_FEC_breakdown(F, sys, FEC_factors, lifetime_years=F.S401.lifetime)
    
    # ED unit의 전체 CAPEX 합계를 계산한 후, 그 1%를 maintenance 항목으로 추가
    total_ed_capex = sum(ed_capex.values())
    ed_opex["Maintenance"] = total_ed_capex * 0.01
    
    # AC 탱크 (ID: T302) CAPEX/OPEX breakdown 추가 (tea 모듈 수정 없이)
    ac_unit = next((u for u in sys.units if u.ID=='T302'), None)
    if ac_unit is not None and hasattr(ac_unit, 'baseline_purchase_costs'):
        ac_capex = ac_unit.baseline_purchase_costs
        total_ac_capex = sum(ac_capex.values())
        # OPEX: 각 구성요소 비용을 10으로 나눈 값 + 전체 CAPEX의 1%를 maintenance 비용으로 추가
        ac_opex = {comp: cost / 20 for comp, cost in ac_capex.items()}
        ac_opex["Maintenance"] = total_ac_capex * 0.01
        if hasattr(ac_unit, 'power_utility'):
            ac_opex["Electricity"] = ac_unit.power_utility.cost * sys.operating_hours
    else:
        ac_capex, ac_opex = {}, {}

    # DC Tank (ID: T301) CAPEX/OPEX breakdown 추가
    dc = next((u for u in sys.units if u.ID=='T301'), None)
    if dc is not None and hasattr(dc, 'baseline_purchase_costs'):
        dc_capex = dc.baseline_purchase_costs
        total_dc_capex = sum(dc_capex.values())
        # OPEX: 각 구성요소 비용을 10으로 나눈 값 + 전체 CAPEX의 1% (maintenance) 추가
        dc_opex = {comp: cost / 20 for comp, cost in dc_capex.items()}
        dc_opex["Maintenance"] = total_dc_capex * 0.01
        if hasattr(dc, 'power_utility'):
            dc_opex["Electricity"] = dc.power_utility.cost * sys.operating_hours
    else:
        dc_capex, dc_opex = {}, {}

    # UASB, CellMassFilter, BatchCrystallizer, DrumDryer CAPEX/OPEX 추출 (ID: R101, U302, S201, D301)
    # UASB (ID: R101)
    uasb = next((u for u in sys.units if u.ID=='R101'), None)
    if uasb is not None and hasattr(uasb, 'baseline_purchase_costs'):
        uasb_capex = uasb.baseline_purchase_costs
        total_uasb_capex = sum(uasb_capex.values())
        # OPEX: 각 구성요소 비용을 20으로 나눈 값 + 전체 CAPEX의 1%를 유지보수 비용으로 추가
        uasb_opex = {comp: cost / 20 for comp, cost in uasb_capex.items()}
        maintenance_cost = total_uasb_capex * 0.01
        uasb_opex["Maintenance"] = maintenance_cost
        if hasattr(uasb, 'power_utility'):
            uasb_opex["Electricity"] = uasb.power_utility.cost * sys.operating_hours
    else:
        uasb_capex, uasb_opex = {}, {}
    
    # CellMassFilter (ID: U302)
    cell_mass = next((u for u in sys.units if u.ID=='U302'), None)
    if cell_mass is not None and hasattr(cell_mass, 'baseline_purchase_costs'):
        cell_mass_capex = cell_mass.baseline_purchase_costs
        total_cell_mass_capex = sum(cell_mass_capex.values())
        cell_mass_opex = {comp: cost / 20 for comp, cost in cell_mass_capex.items()}
        maintenance_cost = total_cell_mass_capex * 0.01
        cell_mass_opex["Maintenance"] = maintenance_cost
        if hasattr(cell_mass, 'power_utility'):
            cell_mass_opex["Electricity"] = cell_mass.power_utility.cost * sys.operating_hours
    else:
        cell_mass_capex, cell_mass_opex = {}, {}
    
    # BatchCrystallizer (ID: S201)
    batch_cryst = next((u for u in sys.units if u.ID=='S201'), None)
    if batch_cryst is not None and hasattr(batch_cryst, 'baseline_purchase_costs'):
        batch_cryst_capex = batch_cryst.baseline_purchase_costs
        total_batch_cryst_capex = sum(batch_cryst_capex.values())
        batch_cryst_opex = {comp: cost / 20 for comp, cost in batch_cryst_capex.items()}
        maintenance_cost = total_batch_cryst_capex * 0.01
        batch_cryst_opex["Maintenance"] = maintenance_cost
        if hasattr(batch_cryst, 'power_utility'):
            batch_cryst_opex["Electricity"] = batch_cryst.power_utility.cost * sys.operating_hours
    else:
        batch_cryst_capex, batch_cryst_opex = {}, {}
    
    # DrumDryer (ID: D301)
    drum_dryer = next((u for u in sys.units if u.ID=='D301'), None)
    if drum_dryer is not None and hasattr(drum_dryer, 'baseline_purchase_costs'):
        drum_dryer_capex = drum_dryer.baseline_purchase_costs
        total_drum_dryer_capex = sum(drum_dryer_capex.values())
        drum_dryer_opex = {comp: cost / 20 for comp, cost in drum_dryer_capex.items()}
        maintenance_cost = total_drum_dryer_capex * 0.01
        drum_dryer_opex["Maintenance"] = maintenance_cost
        if hasattr(drum_dryer, 'power_utility'):
            drum_dryer_opex["Electricity"] = drum_dryer.power_utility.cost * sys.operating_hours
    else:
        drum_dryer_capex, drum_dryer_opex = {}, {}


    # 단위를 백만 달러 단위로 변환 (CAPEX/OPEX에만 해당)
    ed_capex_millions = {k: v / 1e6 for k, v in ed_capex.items()}
    ed_opex_millions = {k: v / 1e6 for k, v in ed_opex.items()}
    ac_capex_millions = {k: v / 1e6 for k, v in ac_capex.items()}
    ac_opex_millions = {k: v / 1e6 for k, v in ac_opex.items()}
    # 단위를 백만 달러 단위로 변환 (CAPEX/OPEX에만 해당)
    dc_capex_millions = {k: v / 1e6 for k, v in dc_capex.items()}
    dc_opex_millions = {k: v / 1e6 for k, v in dc_opex.items()}

    # 추가 유닛 단위 변환
    uasb_capex_millions = {k: v / 1e6 for k, v in uasb_capex.items()}
    uasb_opex_millions = {k: v / 1e6 for k, v in uasb_opex.items()}
    cell_mass_capex_millions = {k: v / 1e6 for k, v in cell_mass_capex.items()}
    cell_mass_opex_millions = {k: v / 1e6 for k, v in cell_mass_opex.items()}
    batch_cryst_capex_millions = {k: v / 1e6 for k, v in batch_cryst_capex.items()}
    batch_cryst_opex_millions = {k: v / 1e6 for k, v in batch_cryst_opex.items()}
    drum_dryer_capex_millions = {k: v / 1e6 for k, v in drum_dryer_capex.items()}
    drum_dryer_opex_millions = {k: v / 1e6 for k, v in drum_dryer_opex.items()}
    
    # DataFrame 생성
    ed_capex_df = pd.DataFrame(list(ed_capex_millions.items()), columns=["Component", "Cost (Million USD)"])
    ed_opex_df = pd.DataFrame(list(ed_opex_millions.items()), columns=["Component", "Cost (Million USD/yr)"])
    ac_capex_df = pd.DataFrame(list(ac_capex_millions.items()), columns=["Component", "Cost (Million USD)"])
    ac_opex_df = pd.DataFrame(list(ac_opex_millions.items()), columns=["Component", "Cost (Million USD/yr)"])
    dc_capex_df = pd.DataFrame(list(dc_capex_millions.items()), columns=["Component", "CAPEX (Million USD)"])
    dc_opex_df = pd.DataFrame(list(dc_opex_millions.items()), columns=["Component", "OPEX (Million USD/yr)"])
    ed_gwp_df = pd.DataFrame(list(ed_gwp_breakdown.items()), columns=["Component", "GWP (kg CO2-eq)"])
    ed_fec_df = pd.DataFrame(list(ed_fec_breakdown.items()), columns=["Component", "FEC (kg oil-eq)"])
    
    uasb_capex_df = pd.DataFrame(list(uasb_capex_millions.items()), columns=["Component", "CAPEX (Million USD)"])
    uasb_opex_df = pd.DataFrame(list(uasb_opex_millions.items()), columns=["Component", "OPEX (Million USD/yr)"])
    cell_mass_capex_df = pd.DataFrame(list(cell_mass_capex_millions.items()), columns=["Component", "CAPEX (Million USD)"])
    cell_mass_opex_df = pd.DataFrame(list(cell_mass_opex_millions.items()), columns=["Component", "OPEX (Million USD/yr)"])
    batch_cryst_capex_df = pd.DataFrame(list(batch_cryst_capex_millions.items()), columns=["Component", "CAPEX (Million USD)"])
    batch_cryst_opex_df = pd.DataFrame(list(batch_cryst_opex_millions.items()), columns=["Component", "OPEX (Million USD/yr)"])
    drum_dryer_capex_df = pd.DataFrame(list(drum_dryer_capex_millions.items()), columns=["Component", "CAPEX (Million USD)"])
    drum_dryer_opex_df = pd.DataFrame(list(drum_dryer_opex_millions.items()), columns=["Component", "OPEX (Million USD/yr)"])


    # Excel 파일로 저장 (여러 시트)
    with pd.ExcelWriter("breakdown_results.xlsx") as writer:
        ed_capex_df.to_excel(writer, sheet_name="ED_CAPEX", index=False)
        ed_opex_df.to_excel(writer, sheet_name="ED_OPEX", index=False)
        ac_capex_df.to_excel(writer, sheet_name="AC_CAPEX", index=False)
        ac_opex_df.to_excel(writer, sheet_name="AC_OPEX", index=False)
        dc_capex_df.to_excel(writer, sheet_name="DC_CAPEX", index=False)       # DC 탱크 CAPEX 시트 추가
        dc_opex_df.to_excel(writer, sheet_name="DC_OPEX", index=False)         # DC 탱크 OPEX 시트 추가
        ed_gwp_df.to_excel(writer, sheet_name="ED+Tank_GWP", index=False)
        ed_fec_df.to_excel(writer, sheet_name="ED+Tank_FEC", index=False)
        uasb_capex_df.to_excel(writer, sheet_name="UASB_CAPEX", index=False)
        uasb_opex_df.to_excel(writer, sheet_name="UASB_OPEX", index=False)
        cell_mass_capex_df.to_excel(writer, sheet_name="CellMass_CAPEX", index=False)
        cell_mass_opex_df.to_excel(writer, sheet_name="CellMass_OPEX", index=False)
        batch_cryst_capex_df.to_excel(writer, sheet_name="BatchCryst_CAPEX", index=False)
        batch_cryst_opex_df.to_excel(writer, sheet_name="BatchCryst_OPEX", index=False)
        drum_dryer_capex_df.to_excel(writer, sheet_name="DrumDryer_CAPEX", index=False)
        drum_dryer_opex_df.to_excel(writer, sheet_name="DrumDryer_OPEX", index=False)
        # 기존 파라메트릭 스윕 결과 시트도 추가
        sweep_df.to_excel(writer, sheet_name="CurrentDensityvsMembraneArea", index=False)

    print("Breakdown results and parametric sweep data saved to breakdown_results.xlsx")