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
import thermosteam as tmo
from biorefineries.cornstover import CellulosicEthanolTEA


class VFA_TEA(CellulosicEthanolTEA):
    """
    Techno-Economic Analysis (TEA) for VFA production and separation.
    This class calculates CAPEX, OPEX, and other economic indicators.
    """
    __slots__ = ('OSBL_units', 'warehouse', 'site_development',
                 'additional_piping', 'proratable_costs', 'field_expenses',
                 'construction', 'contingency', 'other_indirect_costs',
                 'labor_cost', 'labor_burden', 'property_insurance',
                 'maintenance', '_ISBL_DPI_cached', '_FCI_cached',
                 '_utility_cost_cached', '_DPI_cached', '_TDC_cached')

    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)
        self.OSBL_units = kwargs.get('OSBL_units', None)
        self.warehouse = kwargs.get('warehouse', 0.04)
        self.site_development = kwargs.get('site_development', 0.09)
        self.additional_piping = kwargs.get('additional_piping', 0.045)
        self.proratable_costs = kwargs.get('proratable_costs', 0.10)
        self.field_expenses = kwargs.get('field_expenses', 0.10)
        self.construction = kwargs.get('construction', 0.20)
        self.contingency = kwargs.get('contingency', 0.10)
        self.other_indirect_costs = kwargs.get('other_indirect_costs', 0.10)
        self.labor_cost = kwargs.get('labor_cost', 2.5e6)
        self.labor_burden = kwargs.get('labor_burden', 0.90)
        self.property_insurance = kwargs.get('property_insurance', 0.007)
        self.maintenance = kwargs.get('maintenance', 0.03)
        self.steam_power_depreciation = kwargs.get('steam_power_depreciation', 'MACRS20')
        self.boiler_turbogenerator = kwargs.get('boiler_turbogenerator', None)

    @property
    def CAPEX(self):
        """
        Calculate the Total Capital Investment (TCI), which includes:
        - ISBL costs
        - OSBL costs
        - Indirect costs (e.g., contingency, site development)
        - Working capital
        """
        DPI = self.DPI
        TDC = self._TDC(DPI, self.installed_equipment_cost)
        working_capital = self.WC_over_FCI * TDC
        TCI = TDC + working_capital
        return TCI

    @property
    def OPEX(self):
        """
        Calculate the Total Operating Costs (OPEX), which includes:
        - Fixed Operating Costs (FOC)
        - Variable Operating Costs (VOC)
        """
        FCI = self.FCI
        FOC = self._FOC(FCI)
        VOC = self.VOC
        return FOC + VOC

    def _TDC(self, DPI, installed_equipment_cost):
        """
        Calculate the Total Depreciable Capital (TDC).
        """
        if installed_equipment_cost is None:
            installed_equipment_cost = self.installed_equipment_cost  # 시스템에서 가져오기
        indirect_costs = self._depreciable_indirect_costs(installed_equipment_cost)
        TDC = DPI + indirect_costs
        self._TDC_cached = TDC
        return TDC

    def _depreciable_indirect_costs(self, installed_equipment_cost):
        """
        Calculate depreciable indirect costs such as contingency and field expenses.
        """
        return (self.proratable_costs + self.field_expenses
                + self.construction + self.contingency) * self._ISBL_DPI(installed_equipment_cost)

    def _FOC(self, FCI):
        """
        Calculate Fixed Operating Costs (FOC).
        """
        ISBL = self._ISBL_DPI_cached
        return (FCI * self.property_insurance  # Property insurance
                + ISBL * self.maintenance  # Maintenance
                + self.labor_cost * (1 + self.labor_burden))  # Labor and burden

    def generate_report(self):
        """
        Generate a CAPEX and OPEX report for the VFA process.
        """
        print("----- VFA TEA Report -----")
        print(f"Total Capital Investment (CAPEX): ${self.CAPEX:,.2f}")
        print(f"Total Operating Costs (OPEX): ${self.OPEX:,.2f}")
        print(f"Variable Operating Costs (VOC): ${self.VOC:,.2f}")
        print(f"Fixed Operating Costs (FOC): ${self._FOC(self.FCI):,.2f}")
        print("--------------------------")

def create_vfa_tea(system, **kwargs):
    """
    Factory function to create a VFA TEA instance with default parameters.
    """
    OSBL_units = kwargs.get('OSBL_units', bst.get_OSBL(system.cost_units))
    # boiler_turbogenerator = kwargs.get(
    #     'boiler_turbogenerator',
    #     tmo.utils.get_instance(OSBL_units, (bst.BoilerTurbogenerator, bst.Boiler))
    # )
    installed_equipment_cost = kwargs.get('installed_equipment_cost', system.installed_cost)  # 시스템에서 비용 가져오기
    vfa_tea = VFA_TEA(
        system=system,
        IRR=kwargs.get('IRR', 0.10),
        duration=kwargs.get('duration', (2023, 2043)),
        depreciation=kwargs.get('depreciation', 'MACRS7'),
        income_tax=kwargs.get('income_tax', 0.21),
        operating_days=kwargs.get('operating_days', 350),
        lang_factor=kwargs.get('lang_factor', None),
        construction_schedule=kwargs.get('construction_schedule', (0.08, 0.6, 0.32)),
        startup_months=kwargs.get('startup_months', 3),
        startup_FOCfrac=kwargs.get('startup_FOCfrac', 1),
        startup_VOCfrac=kwargs.get('startup_VOCfrac', 0.75),
        startup_salesfrac=kwargs.get('startup_salesfrac', 0.5),
        WC_over_FCI=kwargs.get('WC_over_FCI', 0.05),
        finance_interest=kwargs.get('finance_interest', 0.08),
        finance_years=kwargs.get('finance_years', 10),
        finance_fraction=kwargs.get('finance_fraction', 0.4),
        OSBL_units=OSBL_units,
        warehouse=kwargs.get('warehouse', 0.04),  # 명시적으로 기본값 설정
        site_development=kwargs.get('site_development', 0.09),
        additional_piping=kwargs.get('additional_piping', 0.045),
        proratable_costs=kwargs.get('proratable_costs', 0.10),
        field_expenses=kwargs.get('field_expenses', 0.10),
        construction=kwargs.get('construction', 0.20),
        contingency=kwargs.get('contingency', 0.10),
        other_indirect_costs=kwargs.get('other_indirect_costs', 0.10),
        labor_cost=kwargs.get('labor_cost', 2.5e6),
        labor_burden=kwargs.get('labor_burden', 0.90),
        property_insurance=kwargs.get('property_insurance', 0.007),
        maintenance=kwargs.get('maintenance', 0.03),
        steam_power_depreciation=kwargs.get('steam_power_depreciation', 'MACRS20'),
        boiler_turbogenerator=None
    )
    vfa_tea.installed_equipment_cost = installed_equipment_cost  # 시스템에서 가져온 값을 저장
    return vfa_tea
