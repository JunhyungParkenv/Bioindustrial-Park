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
#%% 📌 **1. Membrane Area vs. Current Density 분석**
j_values = np.linspace(5, 30, 25)  # 5 ~ 30 A/m² 범위에서 25개 샘플
A_m_values = []

# ✅ 결과 저장용 DataFrame
df_j_vs_A_m = pd.DataFrame(columns=['Current Density (A/m²)', 'Membrane Area (m²)'])

for j in j_values:
    # ✅ Flowsheet 초기화 (중복 방지)
    bst.main_flowsheet.clear()
    
    # ✅ 시스템 생성
    VFA_sys = create_VFA_sys()
    ED_unit = VFA_sys.flowsheet.unit.S401  # Electrodialysis 유닛 가져오기
    
    # ✅ 전류 밀도 업데이트
    ED_unit.j = j
    
    # ✅ 시뮬레이션 실행
    VFA_sys.simulate()
    
    # ✅ 업데이트된 Membrane Area 저장
    A_m = ED_unit.A_m
    A_m_values.append(A_m)
    
    # ✅ DataFrame 업데이트
    new_row = pd.DataFrame({'Current Density (A/m²)': [float(j)], 'Membrane Area (m²)': [float(A_m)]})
    df_j_vs_A_m = pd.concat([df_j_vs_A_m, new_row], ignore_index=True)
#%% 📌 **2. Tau vs. Tank Volume, Membrane Area, Current Density 분석**
tau_values = np.linspace(0.1, 20, 25)  # 0.1시간 ~ 20시간 (25개 샘플)
tank_volumes = []
membrane_areas = []
current_densities = []

# ✅ 결과 저장용 DataFrame
df_tau_vs_A_m = pd.DataFrame(columns=['Tau (hr)', 'Tank Volume (m³)', 'Membrane Area (m²)', 'Current Density (A/m²)'])

for tau in tau_values:
    # ✅ Flowsheet 초기화 (중복 방지)
    bst.main_flowsheet.clear()
    
    # ✅ 시스템 생성
    VFA_sys = create_VFA_sys()
    AC_tank = VFA_sys.flowsheet.unit.ac_tank  # AC Storage Tank
    ED_unit = VFA_sys.flowsheet.unit.S401  # Electrodialysis Unit (ED)
    
    # ✅ 체류시간 (tau) 업데이트
    AC_tank.tau = tau
    
    # ✅ 시뮬레이션 실행
    VFA_sys.simulate()
    
    # ✅ 업데이트된 값 가져오기
    tank_volume = AC_tank.design_results['Volume']
    membrane_area = ED_unit.A_m
    current_density = ED_unit.j

    # ✅ 리스트에 저장
    tank_volumes.append(tank_volume)
    membrane_areas.append(membrane_area)
    current_densities.append(current_density)
    
    # ✅ DataFrame 업데이트
    new_row = pd.DataFrame({'Tau (hr)': [tau], 'Tank Volume (m³)': [tank_volume], 
                            'Membrane Area (m²)': [membrane_area], 'Current Density (A/m²)': [current_density]})
    df_tau_vs_A_m = pd.concat([df_tau_vs_A_m, new_row], ignore_index=True)

# 📌 **3. 결과 출력**
print("\n🔹 **Membrane Area vs. Current Density** 🔹")
print(df_j_vs_A_m.to_string(index=False))  # 전류 밀도 vs. Membrane Area 테이블 출력

print("\n🔹 **AC Storage Tank Tau vs. System Parameters** 🔹")
print(df_tau_vs_A_m.to_string(index=False))  # Tau vs. 시스템 파라미터 테이블 출력

# 📌 **4. 그래프 출력**
plt.figure(figsize=(12, 8))
#%% ✅ (1) Membrane Area vs. Current Density
plt.subplot(2, 2, 1)
plt.plot(j_values, A_m_values, 'r-o', alpha=0.7, label="Membrane Area")
plt.xlabel("Current Density (j) [A/m²]")
plt.ylabel("Membrane Area (A_m) [m²]")
plt.title("Membrane Area vs. Current Density")
plt.grid(True)
plt.legend()

# ✅ (2) Tau vs. Tank Volume
plt.subplot(2, 2, 2)
plt.plot(tau_values, tank_volumes, 'b-o', alpha=0.7, label="Tank Volume")
plt.xlabel("Tau (hr)")
plt.ylabel("Tank Volume (m³)")
plt.title("Tau vs. Tank Volume")
plt.grid(True)
plt.legend()

# ✅ (3) Tau vs. Membrane Area
plt.subplot(2, 2, 3)
plt.plot(tau_values, membrane_areas, 'g-o', alpha=0.7, label="Membrane Area")
plt.xlabel("Tau (hr)")
plt.ylabel("Membrane Area (m²)")
plt.title("Tau vs. Membrane Area")
plt.grid(True)
plt.legend()

# ✅ (4) Tau vs. Current Density
plt.subplot(2, 2, 4)
plt.plot(tau_values, current_densities, 'm-o', alpha=0.7, label="Current Density")
plt.xlabel("Tau (hr)")
plt.ylabel("Current Density (A/m²)")
plt.title("Tau vs. Current Density")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
#%% 📌 **5. 3D Surface Plot (Tau, j, A_m)**
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_tau_vs_A_m['Tau (hr)'], df_tau_vs_A_m['Current Density (A/m²)'], df_tau_vs_A_m['Membrane Area (m²)'], 
           c=df_tau_vs_A_m['Membrane Area (m²)'], cmap='coolwarm', s=40)

ax.set_xlabel("Tau (hr)")
ax.set_ylabel("Current Density (A/m²)")
ax.set_zlabel("Membrane Area (m²)")
ax.set_title("3D Relationship: Tau vs. j vs. Membrane Area")
plt.show()

# 📌 **6. 최적점 탐색 (Min Membrane Area & Balanced j)**
optimal_index = np.argmin(df_tau_vs_A_m['Membrane Area (m²)'])  # 최소 Membrane Area를 가지는 인덱스
optimal_tau = df_tau_vs_A_m.loc[optimal_index, 'Tau (hr)']
optimal_j = df_tau_vs_A_m.loc[optimal_index, 'Current Density (A/m²)']
optimal_A_m = df_tau_vs_A_m.loc[optimal_index, 'Membrane Area (m²)']

print("\n✅ **Optimal Point Found!** ✅")
print(f"🔹 Optimal Tau: {optimal_tau:.2f} hr")
print(f"🔹 Optimal Current Density (j): {optimal_j:.2f} A/m²")
print(f"🔹 Optimal Membrane Area (A_m): {optimal_A_m:.2f} m²")