# -*- coding: utf-8 -*-
"""
Updated on Feb 10, 2025

@author: Junhyung Park
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosteam as bst
from biorefineries.VFA._systems_VFA import create_VFA_sys

# ✅ Flowsheet 초기화
bst.main_flowsheet.clear()

#%% 📌 **1. Tau vs. Membrane Area & Current Density 관계 분석**
tau_values = np.linspace(0.1, 20, 25)  # 체류시간 범위 설정
membrane_areas = []
current_densities = []

df_tau_vs_params = pd.DataFrame(columns=['Tau (hr)', 'Membrane Area (m²)', 'Current Density (A/m²)'])

for tau in tau_values:
    # ✅ Flowsheet 재설정
    bst.main_flowsheet.clear()
    VFA_sys = create_VFA_sys()
    AC_tank = VFA_sys.flowsheet.unit.ac_tank
    ED_unit = VFA_sys.flowsheet.unit.S401
    
    # ✅ 1️⃣ AC Tank 체류 시간 설정
    AC_tank.tau = tau
    
    # ✅ 2️⃣ ED 유닛 업데이트 (Flux, A_m, j 자동 계산)
    VFA_sys.simulate()
    
    # ✅ 3️⃣ 업데이트된 ED 유닛 값 가져오기
    membrane_area = ED_unit.A_m
    current_density = ED_unit.j

    # ✅ 값 저장
    membrane_areas.append(membrane_area)
    current_densities.append(current_density)

    # ✅ DataFrame 업데이트
    new_row = pd.DataFrame({'Tau (hr)': [tau], 'Membrane Area (m²)': [membrane_area], 'Current Density (A/m²)': [current_density]})
    df_tau_vs_params = pd.concat([df_tau_vs_params, new_row], ignore_index=True)

# ✅ 결과 출력
print("\n🔹 **Tau vs. Membrane Area & Current Density** 🔹")
print(df_tau_vs_params.to_string(index=False))

#%% ✅ 개별 그래프 출력
plt.figure(figsize=(12, 5))

# ✅ (1) Tau vs. Membrane Area
plt.subplot(1, 2, 1)
plt.plot(tau_values, membrane_areas, 'g-o', alpha=0.7)
plt.xlabel("Tau (hr)")
plt.ylabel("Membrane Area (A_m) [m²]")
plt.title("Tau vs. Membrane Area")
plt.grid(True)

# ✅ (2) Tau vs. Current Density
plt.subplot(1, 2, 2)
plt.plot(tau_values, current_densities, 'm-o', alpha=0.7)
plt.xlabel("Tau (hr)")
plt.ylabel("Current Density (j) [A/m²]")
plt.title("Tau vs. Current Density")
plt.grid(True)

plt.tight_layout()
plt.show()
