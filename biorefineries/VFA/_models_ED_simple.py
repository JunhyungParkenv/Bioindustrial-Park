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

# 시스템 실행
VFA_sys = create_VFA_sys()
VFA_sys.simulate()

# ED 유닛 가져오기
ed_unit = bst.main_flowsheet.unit['S401']

# 전류 밀도 범위 설정 (1 A/m² ~ 50 A/m²)
j_values = np.linspace(1, 20, 10)  # 1~50 A/m² 사이에서 20개 샘플링
A_m_values = []

for j in j_values:
    ed_unit.j = j  # 전류 밀도 변경
    VFA_sys.simulate()  # 시스템 재시뮬레이션
    A_m_values.append(ed_unit.A_m)  # 업데이트된 A_m 저장
    print(f"j = {j:.2f} A/m² → A_m = {ed_unit.A_m:.4f} m²")

# 결과 데이터 저장
df = pd.DataFrame({'Current Density (A/m²)': j_values, 'Membrane Area (m²)': A_m_values})

# 그래프 시각화
plt.figure(figsize=(7, 5))
plt.plot(j_values, A_m_values, 'bo-', markersize=5, label="Membrane Area vs. Current Density")
plt.xlabel("Current Density (A/m²)")
plt.ylabel("Membrane Area (m²)")
plt.title("Current Density vs. Membrane Area in ED System")
plt.grid(True)
plt.legend()
plt.show()
#%%
# Membrane Area vs. AC Tank Volume 분석
A_m_values = np.linspace(1, 50, 10)  # 1~50 m² 사이에서 10개 샘플링
ac_tank_volumes = []

for A_m in A_m_values:
    ed_unit.A_m = A_m  # 멤브레인 면적 변경
    VFA_sys.simulate()  # 시스템 재시뮬레이션
    ac_tank = bst.main_flowsheet.unit['ac_tank']
    ac_tank_volumes.append(ac_tank.design_results['Total volume'])  # 업데이트된 AC 탱크 부피 저장
    print(f"A_m = {A_m:.2f} m² → AC Tank Volume = {ac_tank.design_results['Total volume']:.4f} m³")

# 결과 데이터 저장
df2 = pd.DataFrame({'Membrane Area (m²)': A_m_values, 'AC Tank Volume (m³)': ac_tank_volumes})

# 그래프 시각화
plt.figure(figsize=(7, 5))
plt.plot(A_m_values, ac_tank_volumes, 'ro-', markersize=5, label="AC Tank Volume vs. Membrane Area")
plt.xlabel("Membrane Area (m²)")
plt.ylabel("AC Tank Volume (m³)")
plt.title("Membrane Area vs. AC Tank Volume in ED System")
plt.grid(True)
plt.legend()
plt.show()
#%%
# AC Tank Volume vs. Current Density 분석
ac_tank_volumes = []

for j in j_values:
    ed_unit.j = j  # 전류 밀도 변경
    VFA_sys.simulate()  # 시스템 재시뮬레이션
    ac_tank = bst.main_flowsheet.unit['ac_tank']
    ac_tank_volumes.append(ac_tank.design_results['Total volume'])  # 업데이트된 AC 탱크 부피 저장
    print(f"j = {j:.2f} A/m² → AC Tank Volume = {ac_tank.design_results['Total volume']:.4f} m³")

# 결과 데이터 저장
df3 = pd.DataFrame({'Current Density (A/m²)': j_values, 'AC Tank Volume (m³)': ac_tank_volumes})

# 그래프 시각화
plt.figure(figsize=(7, 5))
plt.plot(j_values, ac_tank_volumes, 'go-', markersize=5, label="AC Tank Volume vs. Current Density")
plt.xlabel("Current Density (A/m²)")
plt.ylabel("AC Tank Volume (m³)")
plt.title("Current Density vs. AC Tank Volume in ED System")
plt.grid(True)
plt.legend()
plt.show()
#%%
# Contour Plot: Current Density (X) vs. AC Tank Volume (Y) with Membrane Area (Color)
# 전류 밀도와 AC 탱크 부피의 범위를 조정하여 더 조밀하게 샘플링
j_values = np.linspace(1, 15, 15)  # 1~10 A/m² 사이에서 15개 샘플링
ac_tank_volumes = np.linspace(0.01, 500, 15)  # 0.01~1 m³ 사이에서 15개 샘플링

J, AC_Vol = np.meshgrid(j_values, ac_tank_volumes)
Membrane_Area = np.zeros_like(J)

for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        ed_unit.j = J[i, j]  # 전류 밀도 업데이트
        ac_tank.design_results['Total volume'] = AC_Vol[i, j]  # AC Tank Volume 업데이트
        VFA_sys.simulate()  # 시스템 재시뮬레이션
        Membrane_Area[i, j] = ed_unit.A_m  # 업데이트된 Membrane Area 저장
        print(f"j = {J[i, j]:.2f} A/m², AC Tank Volume = {AC_Vol[i, j]:.2f} m³ → A_m = {Membrane_Area[i, j]:.4f} m²")

# Contour plot 생성
plt.figure(figsize=(7, 5))
contour = plt.contourf(J, AC_Vol, Membrane_Area, cmap="viridis", levels=20)
cbar = plt.colorbar(contour)
cbar.set_label("Membrane Area (m²)")
plt.xlabel("Current Density (A/m²)")
plt.ylabel("AC Tank Volume (m³)")
plt.title("Optimized Membrane Area Distribution in ED System")
plt.grid(True)
plt.show()
#%%
# 1. 전류 밀도(A/m²) vs. 전력 소비량(W)
j_values = np.linspace(1, 20, 10)  
power_consumption = []

for j in j_values:
    ed_unit.j = j  
    VFA_sys.simulate()  
    power_consumption.append(ed_unit.design_results['Power consumption'])  
    print(f"j = {j:.2f} A/m² → Power Consumption = {ed_unit.design_results['Power consumption']:.2f} W")

plt.figure(figsize=(7, 5))
plt.plot(j_values, power_consumption, 'mo-', markersize=5, label="Power Consumption vs. Current Density")
plt.xlabel("Current Density (A/m²)")
plt.ylabel("Power Consumption (W)")
plt.title("Current Density vs. Power Consumption in ED System")
plt.grid(True)
plt.legend()
plt.show()
#%%
# 2. 멤브레인 면적(m²) vs. VFA 처리량(kg/hr)
A_m_values = np.linspace(1, 50, 10)  
VFA_output_mass = []

for A_m in A_m_values:
    ed_unit.A_m = A_m  
    VFA_sys.simulate()  
    eff_ac = ed_unit.outs[1]  
    total_vfa_mass = eff_ac.imass['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid'].sum()  
    VFA_output_mass.append(total_vfa_mass)  
    print(f"A_m = {A_m:.2f} m² → VFA Output = {total_vfa_mass:.2f} kg/hr")

plt.figure(figsize=(7, 5))
plt.plot(A_m_values, VFA_output_mass, 'c^-', markersize=5, label="VFA Output vs. Membrane Area")
plt.xlabel("Membrane Area (m²)")
plt.ylabel("VFA Output (kg/hr)")
plt.title("Membrane Area vs. VFA Output in ED System")
plt.grid(True)
plt.legend()
plt.show()
#%%
# 3. AC Tank 체류 시간(hr) vs. 목표 농도(g/L) 변화
ac_tau_values = np.linspace(1, 24, 10)
ac_concentration = []

for tau in ac_tau_values:
    ac_tank.tau = tau  
    VFA_sys.simulate()  
    eff_ac = ac_tank.outs[0]  
    total_vfa_mass = eff_ac.imass['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid'].sum()
    solution_volume = eff_ac.F_vol  
    concentration = (total_vfa_mass / solution_volume) if solution_volume > 1e-6 else 0  
    ac_concentration.append(concentration)
    print(f"AC Tank tau = {tau:.2f} hr → Concentration = {concentration:.2f} g/L")

plt.figure(figsize=(7, 5))
plt.plot(ac_tau_values, ac_concentration, 'bs-', markersize=5, label="AC Concentration vs. Residence Time")
plt.xlabel("AC Tank Residence Time (hr)")
plt.ylabel("VFA Concentration (g/L)")
plt.title("AC Tank Residence Time vs. VFA Concentration")
plt.grid(True)
plt.legend()
plt.show()
#%%
# 4. 전류 밀도(A/m²) vs. 이온 플럭스(mol/m²/s) - 개별 이온 분석
flux_results = {ion: [] for ion in ed_unit.CE_dict.keys()}

for j in j_values:
    ed_unit.j = j  
    VFA_sys.simulate()  
    I = ed_unit.j * ed_unit.A_m  
    flux_dict = ed_unit.calculate_flux(I)  

    for ion, flux in flux_dict.items():
        flux_results[ion].append(flux)

plt.figure(figsize=(7, 5))
for ion, flux_list in flux_results.items():
    plt.plot(j_values, flux_list, marker='o', linestyle='-', label=f"{ion} Flux")
plt.xlabel("Current Density (A/m²)")
plt.ylabel("Ion Flux (mol/m²/s)")
plt.title("Current Density vs. Ion Flux in ED System")
plt.grid(True)
plt.legend()
plt.show()
#%%
# 5. 전력 소비량(W) vs. VFA 처리량(kg/hr) - 에너지 효율 분석
power_efficiency = []

for j in j_values:
    ed_unit.j = j  
    VFA_sys.simulate()  
    power = ed_unit.design_results['Power consumption']  
    eff_ac = ed_unit.outs[1]  
    total_vfa_mass = eff_ac.imass['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid'].sum()  
    efficiency = total_vfa_mass / power if power > 0 else 0  
    power_efficiency.append(efficiency)
    print(f"Power = {power:.2f} W → VFA Processing Efficiency = {efficiency:.6f} kg/W")

plt.figure(figsize=(7, 5))
plt.plot(power_consumption, power_efficiency, 'r*-', markersize=5, label="VFA Processing Efficiency")
plt.xlabel("Power Consumption (W)")
plt.ylabel("VFA Processing Efficiency (kg/W)")
plt.title("Energy Efficiency of VFA Processing in ED System")
plt.grid(True)
plt.legend()
plt.show()
#%%
# 🔹 Recycle Ratio (재순환 비율) vs. AC Tank 체류 시간 (`tau`)
recycle_ratios = np.linspace(0.2, 0.9, 8)  # 0% ~ 100% (완전한 배출 ~ 완전 재순환)
ac_tau_values = []

for ratio in recycle_ratios:
    # ✅ ED 유닛의 재순환 비율 업데이트
    recycle_splitter = bst.main_flowsheet.unit['S_AC']  # AC output splitter
    recycle_splitter.split[0] = ratio  # 첫 번째 출력(recycle_ac)의 비율 설정
    
    VFA_sys.simulate()  # 시스템 업데이트
    
    # ✅ AC Tank에서 현재 체류 시간 (`tau`) 가져오기
    ac_tau_values.append(ac_tank.tau)
    print(f"Recycle Ratio = {ratio:.2f} → AC Tank tau = {ac_tank.tau:.4f} hr")

# 🔹 결과 시각화
plt.figure(figsize=(7, 5))
plt.plot(recycle_ratios * 100, ac_tau_values, 'ms-', markersize=5, label="AC Tank tau vs. Recycle Ratio")
plt.xlabel("Recycle Ratio (%)")
plt.ylabel("AC Tank Residence Time (tau) (hr)")
plt.title("Effect of Recycle Ratio on AC Tank Residence Time")
plt.grid(True)
plt.legend()
plt.show()
