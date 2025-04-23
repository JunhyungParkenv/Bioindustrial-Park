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
import biosteam as bst
import pandas as pd

class VFA_TEA(CellulosicEthanolTEA):
    _TCI_ratio_cached = 1
    
    """
    VFA Techno-Economic Analysis (TEA) 모델.
    기존 CellulosicEthanolTEA를 확장하여 VFA 공정에 맞게 수정.
    """
    def __init__(self, system, OSBL_units=None, **kwargs):
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
            labor_cost=2.4e6, # =90*0.9*365*24*33.64 more workers than 60 workers in 'Techno-economic analysis for upgrading the biomass-derived ethanol-to-jet blendstocks'(Ling Tao); 33.64 is employment cost of average 2023 from https://data.bls.gov/cgi-bin/srgate
            labor_burden=0.90,
            property_insurance=0.007, 
            maintenance=0.01,
            steam_power_depreciation='MACRS20',
            boiler_turbogenerator=None
        )
        self.OSBL_units = OSBL_units if OSBL_units is not None else []  # None 방지
    
    @property
    def OSBL_installed_equipment_cost(self):
        """ OSBL 설치 비용 계산 (None 방지) """
        if not self.OSBL_units:  # 🛠️ 빈 리스트 처리
            return 0
        if self.lang_factor:
            raise NotImplementedError('lang factor cannot yet be used')
        elif isinstance(self.system, bst.AgileSystem):
            unit_capital_costs = self.system.unit_capital_costs
            return sum([unit_capital_costs[i].installed_cost for i in self.OSBL_units])
        else:
            return sum([i.installed_cost for i in self.OSBL_units])
    
    # CAPEX를 purchase_cost로만 구성을 함, Fe(CN) 추가해야함
    # ED/AC Tank CAPEX를 따로 구성을 하는거지 (purchase cost)
    @property
    def CAPEX(self):
        """ 전체 CAPEX (USD) """
        # return self.installed_equipment_cost # 구매비용 + 설치비용
        return self.purchase_cost # 구매비용
    
    @property
    def annualized_CAPEX(self):
        i = self.IRR
        n = self.duration[1] - self.duration[0]
        crf = (i * (1 + i)**n) / ((1 + i)**n - 1)
        return self.CAPEX * crf
    
    # OPEX에 Membrane/NF/Current Collector 교체 비용을 추가함, Electricity는 추가함
    # ED/AC Tank OPEX를 따로 구성을 하는거지 (Electricity + 교체비용)
    @property
    def OPEX(self):
        """ 전체 연간 운영 비용 (USD/yr) """
        # return (self.labor_cost + self.material_cost + self.maintenance * self.CAPEX + utility_cost)
        base_opex = (self.system.power_utility.cost * self.system.operating_hours +
                     self.CAPEX * (self.property_insurance + self.maintenance) +
                     self.labor_cost * (1 + self.labor_burden))
        # ED 유닛의 연간 교체 비용 계산
        ed_unit = None
        for unit in self.system.units:
            if unit.ID == 'S401':
                ed_unit = unit
                break
        ed_replacement_cost = 0
        if ed_unit is not None and hasattr(ed_unit, 'baseline_purchase_costs'):
            bp = ed_unit.baseline_purchase_costs
            # 각 부품별 수명: CEM: 5년, NF: 5년, Current Collector: 10년
            if 'CEM' in bp:
                ed_replacement_cost += bp['CEM'] / 5
            if 'NF' in bp:
                ed_replacement_cost += bp['NF'] / 5
            if 'Electrode' in bp:
                ed_replacement_cost += bp['Electrode'] / 10
            if 'Current Collector' in bp:
                ed_replacement_cost += bp['Current Collector'] / 10
            if 'Coating Solution' in bp:
                ed_replacement_cost += bp['Coating Solution'] / 5
            if 'Frames' in bp:
                ed_replacement_cost += bp['Frames'] / 20
            if 'Fe(CN)' in bp:
                ed_replacement_cost += bp['Fe(CN)'] / 20
        #     # 필요한 경우, Coating Solution, Frames 등도 포함할 수 있지만 여기서는 생략
        # return base_opex + ed_replacement_cost

        # AC 탱크 (ID: T302)의 연간 교체 비용 (예: 수명이 20년)
        ac_unit = next((u for u in self.system.units if u.ID == 'T302'), None)
        ac_replacement_cost = 0
        if ac_unit is not None and hasattr(ac_unit, 'baseline_purchase_costs'):
            bp_ac = ac_unit.baseline_purchase_costs
            # AC 탱크의 전체 구매비용을 20년으로 나누어 연간 비용으로 계산
            ac_replacement_cost = sum(bp_ac.values()) / 20
    
        # DC 탱크 (ID: T301)의 연간 교체 비용 (예: 수명이 20년)
        dc_unit = next((u for u in self.system.units if u.ID == 'T301'), None)
        dc_replacement_cost = 0
        if dc_unit is not None and hasattr(dc_unit, 'baseline_purchase_costs'):
            bp_dc = dc_unit.baseline_purchase_costs
            dc_replacement_cost = sum(bp_dc.values()) / 20
    
        return base_opex + ed_replacement_cost + ac_replacement_cost + dc_replacement_cost
    
    @property
    def MPSP(self):
        """ 최소 제품 판매 가격 (MPSP) 수정 """
        return super().MPSP  # 10% 가격 증가 반영
    # --- 추가: ED 유닛 비용 breakdown 프로퍼티 ---
    @property
    def ED_CAPEX_breakdown(self):
        """ ED 유닛('S401')의 구성 부품별 CAPEX breakdown 반환 """
        ed_unit = next((u for u in self.system.units if u.ID == 'S401'), None)
        if ed_unit is None or not hasattr(ed_unit, 'baseline_purchase_costs'):
            return {}
        return ed_unit.baseline_purchase_costs

    @property
    def ED_OPEX_breakdown(self):
        """ ED 유닛('S401')의 구성 부품별 연간 운영비용(OPEX) breakdown 반환 """
        ed_unit = next((u for u in self.system.units if u.ID == 'S401'), None)
        if ed_unit is None or not hasattr(ed_unit, 'baseline_purchase_costs'):
            return {}
        bp = ed_unit.baseline_purchase_costs
        breakdown = {}
        if 'CEM' in bp:
            breakdown['CEM'] = bp['CEM'] / 5
        if 'NF' in bp:
            breakdown['NF'] = bp['NF'] / 5
        if 'Coating Solution' in bp:
            breakdown['Coating Solution'] = bp['Coating Solution'] / 5
        if 'Electrode' in bp:
            breakdown['Electrode'] = bp['Electrode'] / 10  # Electrode 교체 비용
        if 'Current Collector' in bp:
            breakdown['Current Collector'] = bp['Current Collector'] / 10
        if 'Frames' in bp:
            breakdown['Frames'] = bp['Frames'] / 20
        # 전기 비용은 F.S401.power_utility.cost를 이용 (단위: USD/hr)
        if ed_unit is not None and hasattr(ed_unit, 'power_utility'):
            ed_elec_cost = ed_unit.power_utility.cost * self.system.operating_hours
            breakdown['Electricity'] = ed_elec_cost
        return breakdown

    # --- 추가: AC 탱크 비용 breakdown 프로퍼티 (예: 'T302') ---
    @property
    def AC_CAPEX_breakdown(self):
        """ AC 탱크('T302')의 구성 부품별 CAPEX breakdown 반환 """
        ac_unit = next((u for u in self.system.units if u.ID == 'T302'), None)
        if ac_unit is None or not hasattr(ac_unit, 'baseline_purchase_costs'):
            return {}
        return ac_unit.baseline_purchase_costs

    @property
    def AC_OPEX_breakdown(self):
        """ AC 탱크('T302')의 구성 부품별 OPEX breakdown 반환 (교체 비용 + 전기 비용 포함)
            (여기서는 전체 CAPEX를 10년 수명으로 나눈 값을 사용)
        """
        ac_unit = next((u for u in self.system.units if u.ID == 'T302'), None)
        bp = {}
        if ac_unit is not None and hasattr(ac_unit, 'baseline_purchase_costs'):
            bp = ac_unit.baseline_purchase_costs
        breakdown = {comp: cost / 20 for comp, cost in bp.items()}
        # AC 탱크에 전기비용 할당: 만약 ac_unit에 power_utility가 있으면 사용하고, 없으면 0 할당
        if ac_unit is not None and hasattr(ac_unit, 'power_utility'):
            ac_elec_cost = ac_unit.power_utility.cost * self.system.operating_hours
        else:
            ac_elec_cost = 0
        breakdown['Electricity'] = ac_elec_cost
        return breakdown

def create_vfa_tea(system, OSBL_units=None):
    """ VFA TEA 객체 생성 """
    return VFA_TEA(system, OSBL_units=OSBL_units)