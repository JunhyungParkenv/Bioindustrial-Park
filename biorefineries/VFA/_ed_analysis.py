# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 02:35:32 2025

@author: Junhyung Park
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# 시스템과 유닛 모듈 import
from biorefineries.VFA._systems_VFA import VFA_sys  # 시스템 파일에서 시스템 가져오기
from biorefineries.VFA._units import ED, DC_Tank, AC_Tank  # 유닛 파일에서 클래스 가져오기


def get_unit_by_id(units, unit_id):
    """
    Search for a unit by its ID in the list of units.

    Parameters:
    -----------
    units : list
        List of units in the system.
    unit_id : str
        ID of the unit to search for.

    Returns:
    --------
    Unit instance if found, else None.
    """
    for unit in units:
        if unit.ID == unit_id:
            return unit
    raise ValueError(f"Unit with ID '{unit_id}' not found in the system.")


def analyze_ed_tank_relationships(ed_unit, dc_tank, ac_tank):
    """
    Analyze relationships between ED unit, DC tank, and AC tank.

    Parameters:
    -----------
    ed_unit : ED
        Electrodialysis unit instance.
    dc_tank : DC_Tank
        Dilute compartment tank instance.
    ac_tank : AC_Tank
        Concentrate compartment tank instance.
    """
    # Define analysis ranges
    j_values = np.linspace(5, 25, 5)  # Current density (A/m²)
    hrt_values = np.linspace(1, 48, 12)  # Hydraulic Retention Time (hours)

    # Results storage
    membrane_areas = []  # Membrane areas for each j value
    dc_tank_volumes = []  # DC tank volumes for each HRT value
    ac_tank_volumes = []  # AC tank volumes for each HRT value
    ac_volume_vs_j = []  # AC tank volume corresponding to each current density

    # Analyze Membrane Area vs. Current Density
    for j in j_values:
        ed_unit.j = j  # Update current density
        I = ed_unit.j * ed_unit.A_m  # Total current (A)
        total_flux = sum(ed_unit.calculate_flux(I).values())  # Total flux (mol/m²·s)

        # Calculate required membrane area based on flux
        total_vfa_to_transfer = ed_unit.target_ratio * sum(
            ed_unit.ins[0].imol[ion] for ion in ed_unit.CE_dict if ion != "LacticAcid"
        )
        A_m = ed_unit.calculate_membrane_area(total_vfa_to_transfer, total_flux)
        membrane_areas.append(A_m)

        # Calculate AC tank volume for a representative HRT
        hrt = 24  # Representative HRT of 24 hours
        Q_dc = dc_tank.ins[0].F_vol * 1000  # DC flow rate in L/hr
        V_dc = Q_dc * hrt / 1000  # Convert to m³
        V_ac = V_dc * (0.2 / 0.8)  # Assuming 20% AC to DC ratio
        ac_volume_vs_j.append((j, V_ac, A_m))

    # Analyze DC/AC Tank Volumes vs. HRT
    for hrt in hrt_values:
        # Calculate DC tank volume based on HRT
        Q_dc = dc_tank.ins[0].F_vol * 1000  # DC flow rate in L/hr
        V_dc = Q_dc * hrt / 1000  # Convert to m³
        dc_tank_volumes.append(V_dc)

        # Calculate AC tank volume based on ratio
        V_ac = V_dc * (0.2 / 0.8)  # Assuming 20% AC to DC ratio
        ac_tank_volumes.append(V_ac)

    # --- Plot Results ---
    # 1. Membrane Area vs. Current Density
    plt.figure(figsize=(8, 5))
    plt.plot(j_values, membrane_areas, marker="o", color="g", label="Membrane Area (A)")
    plt.xlabel("Current Density (j) [A/m²]", fontsize=14, fontweight="bold")
    plt.ylabel("Membrane Area (A) [m²]", fontsize=14, fontweight="bold")
    plt.title("Membrane Area vs. Current Density", fontsize=16, fontweight="bold")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

    # 2. Tank Volumes vs. HRT
    plt.figure(figsize=(10, 6))
    plt.plot(hrt_values, dc_tank_volumes, marker="o", label="DC Tank Volume", color="b")
    plt.plot(hrt_values, ac_tank_volumes, marker="x", label="AC Tank Volume", color="r")
    plt.xlabel("Hydraulic Retention Time (HRT) [hr]", fontsize=14, fontweight="bold")
    plt.ylabel("Tank Volume [m³]", fontsize=14, fontweight="bold")
    plt.title("Tank Volume vs. HRT", fontsize=16, fontweight="bold")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

    # 3. Contour Plot for AC Tank Volume, Current Density, and Membrane Area
    ac_volume_vs_j = np.array(ac_volume_vs_j)
    J, V_ac, A_m = ac_volume_vs_j[:, 0], ac_volume_vs_j[:, 1], ac_volume_vs_j[:, 2]
    
    # Check if V_ac values are constant
    if np.allclose(V_ac, V_ac[0]):
        print("Warning: AC Tank Volume (V_ac) values are almost constant.")
        V_ac = np.linspace(min(V_ac) - 10, max(V_ac) + 10, len(V_ac))  # Add slight variation

    # Create meshgrid for contour
    j_grid = np.linspace(min(J) * 0.5, max(J) * 3, 50)  # x축 범위를 3배로 확장
    v_ac_grid = np.linspace(min(V_ac) * 0.5, max(V_ac) * 3, 50)  # y축 범위를 3배로 확장
    j_mesh, v_ac_mesh = np.meshgrid(j_grid, v_ac_grid)

    # Replace griddata with a simple interpolation
    a_m_mesh = np.zeros_like(j_mesh)
    for i in range(len(j_grid)):
        a_m_mesh[:, i] = np.interp(v_ac_grid, V_ac, A_m)

    # Plot the contour
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(j_mesh, v_ac_mesh, a_m_mesh, cmap="summer", levels=20)  # "summer" 컬러맵 (노-초)
    cbar = plt.colorbar(contour)
    cbar.set_label("Membrane Area [m²]", fontsize=12)

    # Mark the optimization point
    opt_j = J[np.argmin(A_m)]  # Current density at minimum membrane area
    opt_v_ac = V_ac[np.argmin(A_m)]  # AC Tank Volume at minimum membrane area
    opt_a_m = min(A_m)  # Minimum membrane area
    plt.plot(opt_j, opt_v_ac, "ro", markersize=8, label="Optimization Point")
    plt.text(opt_j + 0.5, opt_v_ac, f"Min Membrane Area\n{opt_a_m:.2f} m²", fontsize=12, color="red")

    # Add labels and title
    plt.xlabel("Current Density (j) [A/m²]", fontsize=14, fontweight="bold")
    plt.ylabel("AC Tank Volume [m³]", fontsize=14, fontweight="bold")
    plt.title("AC Tank Volume and Membrane Area vs. Current Density (Contour Plot)", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()




# --- Main Execution ---
if __name__ == "__main__":
    # 시스템에서 유닛 가져오기
    ED1 = get_unit_by_id(VFA_sys.units, 'S401')  # Electrodialysis unit
    DC_Tank1 = get_unit_by_id(VFA_sys.units, 'T301')  # DC Tank
    AC_Tank1 = get_unit_by_id(VFA_sys.units, 'T302')  # AC Tank

    # 분석 함수 실행
    analyze_ed_tank_relationships(ED1, DC_Tank1, AC_Tank1)
