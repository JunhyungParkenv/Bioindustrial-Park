#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Bioindustrial-Park: BioSTEAM's Premier Biorefinery Models and Results
# Copyright (C) 2021-, Sarang Bhagwat <sarangb2@illinois.edu>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.

This module is a modified implementation of modules from the following:
[1]	Bhagwat et al., Sustainable Production of Acrylic Acid via 3-Hydroxypropionic Acid from Lignocellulosic Biomass. ACS Sustainable Chem. Eng. 2021, 9 (49), 16659–16669. https://doi.org/10.1021/acssuschemeng.1c05441
[2]	Li et al., Sustainable Lactic Acid Production from Lignocellulosic Biomass. ACS Sustainable Chem. Eng. 2021, 9 (3), 1341–1351. https://doi.org/10.1021/acssuschemeng.0c08055
[3]	Cortes-Peña et al., BioSTEAM: A Fast and Flexible Platform for the Design, Simulation, and Techno-Economic Analysis of Biorefineries under Uncertainty. ACS Sustainable Chem. Eng. 2020, 8 (8), 3302–3310. https://doi.org/10.1021/acssuschemeng.9b07040

@author: Junhyung Park
"""

import biosteam as bst
from biorefineries.VFA._tea import VFA_TEA
from biorefineries.VFA._process_settings import price

class ED_TEA:
    """
    Techno-Economic Analysis (TEA) for Electrodialysis (ED).
    This class calculates CAPEX, OPEX, and economic performance metrics.
    """
    
    def __init__(self, ed_unit):
        self.ed = ed_unit
        self.tea = VFA_TEA(self.ed)  # 기존 VFA_TEA와 연동 가능

    @property
    def CAPEX(self):
        """Calculate capital expenditures (CAPEX) based on ED costs."""
        return self.ed.installed_cost

    @property
    def OPEX(self):
        """Calculate operational expenditures (OPEX) based on power consumption."""
        power_cost = self.ed.design_results['Power consumption'] * price['electricity']
        maintenance_cost = 0.03 * self.CAPEX
        labor_cost = 1e6
        return power_cost + maintenance_cost + labor_cost

    def get_capex_breakdown(self):
        """Return CAPEX breakdown for Electrodialysis."""
        return {
            'Membrane Cost': 100 * self.ed.A_m,
            'Power Supply Cost': 20 * self.ed.A_m,
            'Electrode Cost': 50 * self.ed.A_m,
            'Frame Cost': 10 * self.ed.A_m,
            'Installation Cost': 0.2 * self.CAPEX
        }

    def get_opex_breakdown(self):
        """Return OPEX breakdown for Electrodialysis."""
        return {
            'Electricity Cost': self.ed.design_results['Power consumption'] * price['electricity'],
            'Maintenance Cost': 0.03 * self.CAPEX,
            'Labor Cost': 1e6
        }

# 🔹 ED_TEA 객체 생성
def create_ed_tea(ed_unit):
    return ED_TEA(ed_unit)