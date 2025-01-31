# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:15:53 2025

@author: Junhyung Park
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:15:53 2025

@author: Junhyung Park
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosteam as bst
from chaospy import distributions as shape
from biosteam.evaluation import Model, Metric
from biorefineries.VFA import _units  # 기존 ED 클래스 사용

# 📌 **1. ED 인스턴스 생성**
ed = _units.ED(
    ID="ED1",
    ins=["inf_dc", "inf_ac"],
    outs=["eff_dc", "eff_ac"],
    j=5.058,  # 초기 전류 밀도 (mA/cm²)
    t=24 * 3600,  # 24시간 운전
    target_ratio=0.8
)

# **📌 2. Membrane Area vs. Current Density 관계 분석**
j_values = np.linspace(1, 15, 10)  # 전류 밀도 범위 설정 (1~15 mA/cm²)
area_results = []

for j in j_values:
    ed.j = j  # 전류 밀도 업데이트
    ed._run()  # ED 프로세스 실행
    area_results.append((j, ed.A_m))  # 결과 저장

df = pd.DataFrame(area_results, columns=["Current Density (mA/cm²)", "Membrane Area (m²)"])

# **📌 그래프 그리기 (seaborn 없이 matplotlib 사용)**
plt.figure(figsize=(8, 5))
plt.plot(df["Current Density (mA/cm²)"], df["Membrane Area (m²)"], marker="o", linestyle="-", color="b", label="Membrane Area")
plt.xlabel("Current Density (mA/cm²)")
plt.ylabel("Membrane Area (m²)")
plt.title("Membrane Area vs. Current Density in ED")
plt.grid(True)
plt.legend()
plt.show()

# **📌 3. ED의 CAPEX & OPEX 계산**
ed._design()  # CAPEX, OPEX 계산 수행
capex = ed.installed_cost  # 설치 비용 (CAPEX)
opex = ed.utility_cost  # 연간 운전 비용 (OPEX)
print(f"✅ CAPEX: {capex:.2f} USD")
print(f"✅ OPEX: {opex:.2f} USD/yr")

# **📌 4. Monte Carlo 기반 불확실성 분석**
def create_ed_model():
    """
    ED 모델 생성 (불확실성 분석 포함)
    """
    system = bst.System('ED_System', path=(ed,))
    
    # 📌 메트릭 정의 (주요 평가 항목)
    metrics = [
        Metric('Membrane Area', lambda: ed.A_m, 'm²'),
        Metric('Current Density', lambda: ed.j, 'A/m²'),
        Metric('Total Current', lambda: ed.j * ed.A_m, 'A'),
        Metric('Power Consumption', lambda: ed.design_results['Power consumption'], 'W'),
        Metric('CAPEX', lambda: ed.installed_cost, 'USD'),
        Metric('OPEX', lambda: ed.utility_cost, 'USD/yr'),
    ]

    model = Model(system, metrics)

    # 📌 전류 밀도 (j) 범위: 3~15 mA/cm²
    j_dist = shape.Uniform(3, 15)

    @model.parameter(name='Current Density', element=ed, kind='coupled', units='mA/cm²', distribution=j_dist)
    def set_current_density(j):
        ed.j = j

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
