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

from biorefineries.cornstover import CellulosicEthanolTEA

class VFA_TEA(CellulosicEthanolTEA):
    """
    VFA Techno-Economic Analysis (TEA) 모델.
    기존 CellulosicEthanolTEA를 확장하여 VFA 공정에 맞게 수정.
    """
    def __init__(self, system, **kwargs):
        super().__init__(
            system=system, 
            IRR=0.10, 
            duration=(2023, 2043),
            depreciation='MACRS7', 
            income_tax=0.21,
            operating_days=350,
            lang_factor=None, 
            construction_schedule=(0.08, 0.60, 0.32),
            startup_months=3, 
            startup_FOCfrac=1,
            startup_salesfrac=0.5,
            startup_VOCfrac=0.75,
            WC_over_FCI=0.05,
            finance_interest=0.08,
            finance_years=10,
            finance_fraction=0.6,
            OSBL_units=kwargs.get('OSBL_units', None),
            warehouse=0.04, 
            site_development=0.09, 
            additional_piping=0.045,
            proratable_costs=0.10,
            field_expenses=0.10,
            construction=0.20,
            contingency=0.10,
            other_indirect_costs=0.10, 
            labor_cost=2.5e6,
            labor_burden=0.90,
            property_insurance=0.007, 
            maintenance=0.03,
            steam_power_depreciation='MACRS20',
            boiler_turbogenerator=None
        )
    
    @property
    def CAPEX(self):
        """ 수정된 CAPEX 계산 """
        return self.installed_equipment_cost + self.installed_equipment_cost * 0.1  # 10% 추가 비용
    
    @property
    def OPEX(self):
        """ 연간 운영 비용 (Fixed + Variable Operating Cost) """
        utility_cost = self.utility_cost if hasattr(self, 'utility_cost') else 0
        return (self.labor_cost + self.maintenance * self.CAPEX + utility_cost)

    @property
    def MPSP(self):
        """ 최소 제품 판매 가격 (MPSP) 수정 """
        return super().MPSP * 1.1  # 10% 가격 증가 반영

def create_vfa_tea(system, OSBL_units=None):
    """ VFA TEA 객체 생성 """
    return VFA_TEA(system, OSBL_units=OSBL_units)
