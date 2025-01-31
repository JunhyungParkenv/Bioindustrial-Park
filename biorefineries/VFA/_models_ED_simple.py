# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:15:53 2025

@author: Junhyung Park
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosteam as bst
from biorefineries.VFA._chemicals import chems
from chaospy import distributions as shape
from biosteam.evaluation import Model, Metric
from biorefineries.VFA import _units  # 기존 ED 클래스 사용
from biorefineries.VFA._systems_VFA import create_VFA_sys

#%% 📌 **1. `VFA_sys`를 생성하여 ED 유닛(S401) 가져오기**
VFA_sys = create_VFA_sys()  # ✅ `VFA_sys`를 실제 시스템 객체로 생성
VFA_sys.simulate()  # ✅ 시뮬레이션 실행하여 스트림 업데이트
VFA_sys.diagram('cluster', number=True)  # ✅ 흐름도 확인

# ✅ **ED 유닛(S401) 가져오기**
S401 = VFA_sys.flowsheet.unit.S401
inf_dc = S401.ins[0]  
inf_ac = S401.ins[1]  
eff_dc = S401.outs[0]  
eff_ac = S401.outs[1]  

S401._run()  # ✅ ED 유닛 실행하여 스트림 업데이트

#%% 📌 **2. Membrane Area vs. Current Density 관계 분석**
j_values = np.linspace(1, 15, 10)  # 전류 밀도 범위 설정 (1~15 mA/cm²)
results = []

for j in j_values:
    S401.j = j  
    S401._run()  # ED 프로세스 실행
    S401._design()  # ✅ 설계 값 업데이트
    results.append((j, S401.A_m, S401.design_results['Total current'], S401.design_results['Power consumption']))

df = pd.DataFrame(results, columns=["Current Density (mA/cm²)", "Membrane Area (m²)", "Total Current (A)", "Power Consumption (W)"])

# 📌 그래프 출력
plt.figure(figsize=(8, 5))
plt.plot(df["Current Density (mA/cm²)"], df["Membrane Area (m²)"], marker="o", linestyle="-", label="Membrane Area")
plt.xlabel("Current Density (mA/cm²)")
plt.ylabel("Membrane Area (m²)")
plt.title("Membrane Area vs. Current Density in ED")
plt.grid(True)
plt.legend()
plt.show()

#%% 📌 **3. ED TEA 계산 추가 (CAPEX, OPEX, MPSP)**
S401._design()  # ✅ ED 설계 업데이트

# ✅ Total Current 및 Power Consumption을 동적으로 다시 계산
Total_Current = S401.j * S401.A_m  # ✅ 전류 밀도 * 멤브레인 면적
Power_Consumption = Total_Current**2 * S401.R  # ✅ I^2 * R

CAPEX = S401.installed_cost  # ✅ 설치 비용
OPEX = S401.utility_cost + (0.03 * CAPEX)  # ✅ OPEX = 유틸리티 비용 + 유지보수 비용(3% CAPEX)
Annualized_CAPEX = CAPEX * 0.1  # ✅ 감가상각 (예: 10년)
Annual_Production = eff_ac.imass["AceticAcid"] * 8760  # ✅ 연간 생산량 (Acetic Acid 기준)

MPSP = (OPEX + Annualized_CAPEX) / Annual_Production  # ✅ 최소 제품 판매 가격 (USD/kg)

#%% 📌 **4. Monte Carlo 기반 불확실성 분석**
def create_ed_model():
    """
    ED 유닛(S401)에 대해 Monte Carlo 불확실성 분석 수행
    """
    system = bst.System('ED_System', path=(S401,))

    # 📌 주요 평가 항목 (Metrics)
    metrics = [
        Metric('Membrane Area', lambda: S401.A_m, 'm²'),
        Metric('Current Density', lambda: S401.j, 'A/m²'),
        Metric('Total Current', lambda: S401.design_results['Total current'], 'A'),
        Metric('Power Consumption', lambda: S401.design_results['Power consumption'], 'W'),
        Metric('CAPEX', lambda: S401.installed_cost, 'USD'),
        Metric('OPEX', lambda: S401.utility_cost + (0.03 * S401.installed_cost), 'USD/yr'),
        Metric('MPSP', lambda: (S401.utility_cost + (0.03 * S401.installed_cost) + (S401.installed_cost * 0.1)) / (eff_ac.imass["AceticAcid"] * 8760), 'USD/kg'),
    ]

    model = Model(system, metrics)

    # 📌 전류 밀도 (j) 범위: 3~15 mA/cm²
    j_dist = shape.Uniform(3, 15)

    @model.parameter(name='Current Density', element=S401, kind='coupled', units='mA/cm²', distribution=j_dist)
    def set_current_density(j):
        S401.j = j
        S401._run()
        S401._design()

    return model

# ✅ Monte Carlo 분석 실행
model = create_ed_model()
N_samples = 100  # 샘플 개수
samples = model.sample(N_samples, rule='L')
model.load_samples(samples)
model.evaluate()

# 📌 결과 저장
model.table.to_excel('ED_MonteCarlo_Results.xlsx')
spearman_rho, p_values = model.spearman_r()
spearman_rho.to_excel('ED_Spearman_Rho.xlsx')
p_values.to_excel('ED_Spearman_P.xlsx')

print("✅ Monte Carlo 분석 완료. 결과를 엑셀 파일로 저장했습니다.")