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
from biorefineries.VFA._tea_ED import ED_TEA
from biorefineries.VFA._systems_VFA import create_VFA_sys
from biorefineries.VFA._process_settings import price

#%%
class ED_Optimized(ED):
    """
    ED 유닛을 상속하여 최적화된 Membrane Area 적용 및 TEA 분석을 포함한 클래스.
    """
    def __init__(self, ID='', ins=None, outs=(), **kwargs):
        super().__init__(ID, ins, outs, **kwargs)  # 기존 ED 초기화 유지
        self.tea = None  # TEA 분석을 위한 속성 추가

    def set_optimal_A_m(self, optimal_A_m):
        """
        최적의 Membrane Area를 적용하는 함수.
        """
        self.A_m = optimal_A_m  # 최적 A_m 업데이트

    def attach_tea(self, tea_model):
        """
        Techno-Economic Analysis (TEA) 모델을 연결하는 함수.
        """
        self.tea = tea_model

    @property
    def CAPEX(self):
        """Calculate capital expenditures (CAPEX) using installed cost."""
        return self.installed_cost

    @property
    def OPEX(self):
        """Calculate operational expenditures (OPEX) based on power consumption."""
        power_cost = self.design_results['Power consumption'] * price['electricity']
        maintenance_cost = 0.03 * self.CAPEX  # Maintenance: 3% of CAPEX
        labor_cost = 1e6  # Assumed fixed labor cost
        return power_cost + maintenance_cost + labor_cost

    def get_capex_breakdown(self):
        """Return CAPEX breakdown for Electrodialysis."""
        return {
            'Membrane Cost': 100 * self.A_m,
            'Power Supply Cost': 20 * self.A_m,
            'Electrode Cost': 50 * self.A_m,
            'Frame Cost': 10 * self.A_m,
            'Installation Cost': 0.2 * self.CAPEX
        }

    def get_opex_breakdown(self):
        """Return OPEX breakdown for Electrodialysis."""
        return {
            'Electricity Cost': self.design_results['Power consumption'] * price['electricity'],
            'Maintenance Cost': 0.03 * self.CAPEX,
            'Labor Cost': 1e6
        }
# =============================================================================
# **Step 1: 최적의 Membrane Area (A_m) 찾기 (inf_dc 기반)**
# =============================================================================

def find_optimal_membrane_area(inf_dc, j_range=(3, 7), target_ratio=0.8):
    """
    Optimize membrane area to achieve the target ratio based on inf_dc composition.
    - inf_dc: 실제 모델의 dilute compartment (DC) 스트림
    - j_range: Current density 범위 (A/m²)
    - target_ratio: 목표 농축 비율 (0~1)
    """
    j_values = np.linspace(*j_range, 50)  # Current density 범위
    A_values = []

    # ✅ 올바른 방식으로 VFA 총량 계산 (Lactic Acid 제외)
    vfa_list = ['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid']
    total_initial_vfa = sum(inf_dc.imol[ion] for ion in vfa_list if ion in inf_dc.chemicals)

    total_vfa_to_transfer = total_initial_vfa * target_ratio

    for j in j_values:
        # 총 전류 계산 (I = j * A)
        I = j * 1  # A_m=1로 가정하여 먼저 계산
        flux_per_m2 = I / (96485.3 * 1)  # 단위 면적당 flux 계산
        total_flux = flux_per_m2  # 전체 시스템에 대한 flux

        # 최적 멤브레인 면적 계산
        A_m = total_vfa_to_transfer / (total_flux * 24 * 3600)  # 하루(24시간) 기준
        A_values.append(A_m)

    return j_values, A_values

# =============================================================================
# **Step 2: ED 모델 및 TEA 분석 실행**
# =============================================================================

def create_ed_model(optimal_A_m, inf_dc):
    """
    Create a BioSTEAM model for the ED process with optimal A_m.
    """
    ed_unit = ED_Optimized('ED_Unit', A_m=optimal_A_m, j=5.058)  # 최적 A_m 적용

    metrics = [
        Metric('CAPEX', lambda: ed_unit.CAPEX / 1e6, 'Million USD'),
        Metric('OPEX', lambda: ed_unit.OPEX / 1e6, 'Million USD/yr'),
        Metric('Membrane Area', lambda: ed_unit.A_m, 'm²'),
        Metric('Flux', lambda: ed_unit.j * ed_unit.A_m / (96485.3 * ed_unit.z_T), 'mol/m²/s'),
        Metric('Current Density', lambda: ed_unit.j, 'A/m²'),
        Metric('Power Consumption', lambda: ed_unit.design_results['Power consumption'], 'W'),
    ]

    model = Model(ed_unit, metrics, exception_hook='raise')
    return model

# =============================================================================
# **Step 3: Monte Carlo & Sensitivity Analysis 실행**
# =============================================================================

def run_ed_analysis(inf_dc, N=500):
    """
    Run sensitivity and uncertainty analysis for the ED process using actual inf_dc.
    """
    # 최적 Membrane Area 계산
    j_values, A_values = find_optimal_membrane_area(inf_dc)
    optimal_A_m = A_values[len(A_values) // 2]  # 중간값 선택

    model = create_ed_model(optimal_A_m, inf_dc)
    np.random.seed(42)
    samples = model.sample(N, 'L')  # Latin Hypercube Sampling
    model.load_samples(samples)
    model.evaluate()

    # 결과 저장
    model.table.to_excel('ed_analysis_results.xlsx')

    # Spearman 상관 분석 (상수 값 제거)
    spearman_rho, p_values = model.spearman_r()
    spearman_rho = spearman_rho.dropna()
    p_values = p_values.dropna()

    spearman_rho.to_excel('ed_spearman_rho.xlsx')
    p_values.to_excel('ed_spearman_p.xlsx')

    return model

# =============================================================================
# **Step 4: 실행 (실제 inf_dc 스트림 활용)**
# =============================================================================

if __name__ == '__main__':
    # VFA 시스템 시뮬레이션 실행
    VFA_sys = create_VFA_sys()
    VFA_sys.simulate()

    # ✅ ED 유닛에서 inf_dc 스트림 가져오기
    ed_unit = next(unit for unit in VFA_sys.units if unit.ID == 'S401')  # 올바른 방식
    inf_dc = ed_unit.ins[0]  # ED 유닛의 첫 번째 입력 스트림

    # ✅ 최적 Membrane Area 찾기
    j_values, A_values = find_optimal_membrane_area(inf_dc)
    optimal_A_m = A_values[len(A_values) // 2]  # 중간값 선택

    # ✅ 최적화된 ED 유닛 생성 및 TEA 분석
    optimized_ed = ED_Optimized('ED_Optimized', ins=ed_unit.ins, outs=ed_unit.outs, A_m=optimal_A_m, j=5.058)

    # ✅ 🔹 `optimized_ed`가 아니라, ID와 입출력 스트림을 전달해야 함
    tea_model = ED_TEA(optimized_ed.ID, optimized_ed.ins, optimized_ed.outs)

    optimized_ed.attach_tea(tea_model)

    # ✅ ED 분석 실행
    model = create_ed_model(optimal_A_m, inf_dc)
    model.evaluate()

    print("ED Analysis Complete. Results saved to Excel.")

