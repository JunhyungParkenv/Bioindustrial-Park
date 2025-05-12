from biorefineries.cornstover import CellulosicEthanolTEA
import biosteam as bst
import numpy as np

class VFA_TEA_MEE(CellulosicEthanolTEA):
    """
    VFA Techno-Economic Analysis (TEA) model focused on Multi-Effect Evaporator (MEE) costs,
    now including annual replacement cost.
    """
    _TCI_ratio_cached = 1

    def __init__(self, system, OSBL_units=None, mee_lifetime=20, **kwargs):
        """
        Parameters
        ----------
        system : biosteam System
        OSBL_units : list
        mee_lifetime : float
            MEE 장치의 수명(년 단위). 이 기간으로 CAPEX를 나눠 연간 교체비용을 계산.
        """
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
            OSBL_units=OSBL_units or [],
            warehouse=0.04,
            site_development=0.09,
            additional_piping=0.045,
            proratable_costs=0.10,
            field_expenses=0.10,
            construction=0.20,
            contingency=0.10,
            other_indirect_costs=0.10,
            labor_cost=2.4e6,
            labor_burden=0.90,
            property_insurance=0.007,
            maintenance=0.01,
            steam_power_depreciation='MACRS20',
            boiler_turbogenerator=None
        )
        self.OSBL_units = OSBL_units or []
        self.mee_lifetime = mee_lifetime  # MEE 장치 수명 (년 단위)

    @property
    def MEE_CAPEX_breakdown(self):
        """CAPEX breakdown for the MEE unit (ID='E401')."""
        # mee = next((u for u in self.system.units if u.ID == 'E401'), None)
        # if mee is None or not hasattr(mee, 'baseline_purchase_costs'):
        #     return {}
        # return mee.baseline_purchase_costs
        mee = next((u for u in self.system.units if u.ID == 'E401'), None)
        if mee is None or not hasattr(mee, 'baseline_purchase_costs'):
            return {}
        # P를 바꾼 뒤에 다시 summary(=design+cost) 실행 → baseline_purchase_costs 갱신
        mee._summary()
        return mee.baseline_purchase_costs

    @property
    def MEE_OPEX_breakdown(self):
        """OPEX breakdown for the MEE unit: steam & electricity."""
        breakdown = {}
        mee = next((u for u in self.system.units if u.ID == 'E401'), None)
        if not mee:
            return breakdown

        # 증기 비용
        if hasattr(mee, 'heat_utilities'):
            utils = mee.heat_utilities
            utils_iter = utils.values() if isinstance(utils, dict) else utils
            total_steam_cost = sum(u.cost for u in utils_iter)
            breakdown['Steam'] = total_steam_cost * self.system.operating_hours

        # 진공펌프 전기비용
        if hasattr(mee, 'vacuum_system'):
            pump = mee.vacuum_system
            if hasattr(pump, 'power'):  # [kW]
                breakdown['Electricity'] = pump.power * self.system.operating_hours

        return breakdown

    @property
    def CAPEX(self):
        """Total CAPEX including MEE unit costs."""
        base_capex = self.purchase_cost
        return base_capex + sum(self.MEE_CAPEX_breakdown.values())

    @property
    def OPEX(self):
        """Total annual OPEX including base + MEE utilities + MEE replacement cost."""
        # 1) 기본 OPEX: labor, maintenance & insurance, 전체 전기비용
        base_opex = (
            self.system.power_utility.cost * self.system.operating_hours +
            self.CAPEX * (self.property_insurance + self.maintenance) +
            self.labor_cost * (1 + self.labor_burden)
        )

        # 2) MEE 증기/전기비용
        mee_utilities = sum(self.MEE_OPEX_breakdown.values())

        # 3) MEE 연간 교체비용 = MEE CAPEX 총액 / mee_lifetime
        replacement_opex = sum(self.MEE_CAPEX_breakdown.values()) / self.mee_lifetime

        return base_opex + mee_utilities + replacement_opex

    @property
    def MPSP(self):
        """Minimum product selling price, inherit default TEA calculation."""
        return super().MPSP


def create_vfa_tea_mee(system, OSBL_units=None, mee_lifetime=20):
    """Instantiate the VFA TEA model for MEE-focused analysis."""
    return VFA_TEA_MEE(system, OSBL_units=OSBL_units, mee_lifetime=mee_lifetime)
