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
from biorefineries.VFA._chemicals import chems
from chaospy import distributions as shape
from biosteam.evaluation import Model, Metric
from biorefineries.VFA import _units  # 기존 ED 클래스 사용
from biorefineries.VFA._systems_VFA import create_VFA_sys
#%%
# 📌 **1. `VFA_sys`를 생성하여 ED 유닛(S401) 가져오기**
VFA_sys = create_VFA_sys()  # ✅ `VFA_sys`를 실제 시스템 객체로 생성

# ✅ **시뮬레이션 실행하여 스트림 업데이트**
VFA_sys.simulate()

# ✅ **VFA_sys의 흐름도를 확인**
VFA_sys.diagram('cluster', number=True)

# ✅ **ED 유닛(S401) 가져오기**
S401 = VFA_sys.flowsheet.unit.S401

# 📌 기존 스트림 가져오기
inf_dc = S401.ins[0]  # 기존 inf_dc 스트림
inf_ac = S401.ins[1]  # 기존 inf_ac 스트림
eff_dc = S401.outs[0]  # 기존 eff_dc 스트림
eff_ac = S401.outs[1]  # 기존 eff_ac 스트림

# ✅ **ED 유닛 실행하여 스트림 업데이트**
S401._run()
#%%
# 📌 **2. Membrane Area vs. Current Density 관계 분석**
j_values = np.linspace(1, 15, 10)  # 전류 밀도 범위 설정 (1~15 mA/cm²)
area_results = []

for j in j_values:
    S401.j = j  # 전류 밀도 업데이트
    S401._run()  # ED 프로세스 실행
    area_results.append((j, S401.A_m))  # 결과 저장

df = pd.DataFrame(area_results, columns=["Current Density (mA/cm²)", "Membrane Area (m²)"])

# 📌 그래프 그리기 (seaborn 없이 matplotlib 사용)
plt.figure(figsize=(8, 5))
plt.plot(df["Current Density (mA/cm²)"], df["Membrane Area (m²)"], marker="o", linestyle="-", color="b", label="Membrane Area")
plt.xlabel("Current Density (mA/cm²)")
plt.ylabel("Membrane Area (m²)")
plt.title("Membrane Area vs. Current Density in ED (Existing System)")
plt.grid(True)
plt.legend()
plt.show()
#%%
# 📌 **3. ED의 CAPEX & OPEX 계산**
S401._design()  # CAPEX, OPEX 계산 수행
capex = S401.installed_cost  # 설치 비용 (CAPEX)
opex = S401.utility_cost  # 연간 운전 비용 (OPEX)
print(f"✅ CAPEX: {capex:.2f} USD")
print(f"✅ OPEX: {opex:.2f} USD/yr")

# 📌 **4. Monte Carlo 기반 불확실성 분석**
def create_ed_model():
    """
    기존 시스템에서 ED 유닛(S401)에 대해 Monte Carlo 불확실성 분석을 수행하는 모델 생성
    """
    system = bst.System('ED_System', path=(S401,))
    
    # 📌 메트릭 정의 (주요 평가 항목)
    metrics = [
        Metric('Membrane Area', lambda: S401.A_m, 'm²'),
        Metric('Current Density', lambda: S401.j, 'A/m²'),
        Metric('Total Current', lambda: S401.j * S401.A_m, 'A'),
        Metric('Power Consumption', lambda: S401.design_results['Power consumption'], 'W'),
        Metric('CAPEX', lambda: S401.installed_cost, 'USD'),
        Metric('OPEX', lambda: S401.utility_cost, 'USD/yr'),
    ]

    model = Model(system, metrics)

    # 📌 전류 밀도 (j) 범위: 3~15 mA/cm²
    j_dist = shape.Uniform(3, 15)

    @model.parameter(name='Current Density', element=S401, kind='coupled', units='mA/cm²', distribution=j_dist)
    def set_current_density(j):
        S401.j = j

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
