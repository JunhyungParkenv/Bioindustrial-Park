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

from biosteam import TEA
import biosteam as bst

class VFA_TEA(TEA):
    """
    심플한 VFA Techno-Economic Analysis (TEA) 모델.
    CAPEX, OPEX, 에너지 비용, MPSP 계산.
    """
    __slots__ = ('labor_cost', 'maintenance', 'property_insurance', 'utility_cost')

    def __init__(self, system, labor_cost=2.5e6, maintenance=0.03, property_insurance=0.007):
        super().__init__(system, IRR=0.10, duration=(2023, 2043), depreciation='MACRS7',
                         income_tax=0.21, operating_days=350, lang_factor=None,
                         construction_schedule=(0.08, 0.6, 0.32), startup_months=3,
                         startup_FOCfrac=1, startup_VOCfrac=0.75, startup_salesfrac=0.5,
                         WC_over_FCI=0.05, finance_interest=0.08, finance_years=10, finance_fraction=0.4)

        self.labor_cost = labor_cost
        self.maintenance = maintenance
        self.property_insurance = property_insurance
        self.utility_cost = 0  # 에너지 비용 기본값

    @property
    def CAPEX(self):
        """ 총 고정 투자 비용 (Fixed Capital Investment, FCI) """
        installed_equipment_cost = self.system.installed_cost
        indirect_costs = installed_equipment_cost * (self.maintenance + self.property_insurance)
        return installed_equipment_cost + indirect_costs

    @property
    def OPEX(self):
        """ 연간 운영 비용 (Fixed + Variable Operating Cost) """
        return self.labor_cost + self.maintenance * self.CAPEX + self.utility_cost

    @property
    def MPSP(self):
        """ 최소 제품 판매 가격 (Minimum Product Selling Price, MPSP) """
        production = sum([s.F_mass for s in self.system.products])  # 전체 제품의 연간 생산량 (kg/yr)
        return self.OPEX / production if production > 0 else float('inf')

    def set_utility_cost(self, cost):
        """ 유틸리티 비용 설정 (예: 전기, 스팀 비용) """
        self.utility_cost = cost

    def report(self):
        """ TEA 결과 요약 """
        print("----- VFA TEA Report -----")
        print(f"Total Capital Investment (CAPEX): ${self.CAPEX:,.2f}")
        print(f"Total Operating Costs (OPEX): ${self.OPEX:,.2f}")
        print(f"Minimum Product Selling Price (MPSP): ${self.MPSP:,.2f} per kg")
        print("--------------------------")

def create_vfa_tea(system):
    """ VFA TEA 객체 생성 """
    return VFA_TEA(system)
