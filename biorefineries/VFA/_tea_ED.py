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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electrodialysis (ED) Techno-Economic Analysis (TEA) Module

Includes:
- CAPEX & OPEX calculations
- CAPEX & OPEX breakdowns
"""

import biosteam as bst
from biorefineries.VFA._tea import VFA_TEA
from biorefineries.VFA._process_settings import price
from biorefineries.VFA._units import ED

class ED_TEA(ED):
    """
    Techno-Economic Analysis (TEA) for Electrodialysis (ED).
    This class calculates CAPEX, OPEX, and economic performance metrics.
    """
    
    def __init__(self, ID='', ins=None, outs=(), **kwargs):
        super().__init__(ID, ins, outs, **kwargs)
        self.tea = VFA_TEA(self)  # 기존 TEA 연동

    @property
    def CAPEX(self):
        """Calculate capital expenditures (CAPEX) using installed cost."""
        return self.installed_cost  # ⚠ sum(self._cost) 대신 사용

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

# 🔹 ED_TEA 객체 생성
def create_ed_tea(ed_unit):
    return ED_TEA(ed_unit)