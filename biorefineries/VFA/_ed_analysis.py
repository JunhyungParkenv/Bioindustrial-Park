# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 02:35:32 2025

@author: Junhyung Park
"""

import numpy as np
import matplotlib.pyplot as plt

# 시스템과 유닛 모듈 import
from biorefineries.VFA._systems_VFA import VFA_sys  # 시스템 파일에서 시스템 가져오기
from biorefineries.VFA._units import ED, DC_Tank, AC_Tank  # 유닛 파일에서 클래스 가져오기

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


# --- Main Execution ---
if __name__ == "__main__":
    # 시스템에서 유닛 가져오기
    ED1 = VFA_sys.units['S401']  # Electrodialysis unit
    DC_Tank1 = VFA_sys.units['T301']  # DC Tank
    AC_Tank1 = VFA_sys.units['T302']  # AC Tank

    # 분석 함수 실행
    analyze_ed_tank_relationships(ED1, DC_Tank1, AC_Tank1)
