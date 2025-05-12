from biorefineries.cornstover import CellulosicEthanolTEA
import biosteam as bst
import numpy as np

class VFA_TEA_MEE(CellulosicEthanolTEA):
    """
    VFA Techno-Economic Analysis (TEA) model focused on Multi-Effect Evaporator (MEE) costs.
    """
    _TCI_ratio_cached = 1

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

    @property
    def MEE_CAPEX_breakdown(self):
        """CAPEX breakdown for the MEE unit (ID='E401')."""
        mee = next((u for u in self.system.units if u.ID == 'E401'), None)
        if mee is None or not hasattr(mee, 'baseline_purchase_costs'):
            return {}
        return mee.baseline_purchase_costs

    @property
    def MEE_OPEX_breakdown(self):
        """OPEX breakdown for the MEE unit: steam & electricity."""
        breakdown = {}
        mee = next((u for u in self.system.units if u.ID == 'E401'), None)
        if mee:
            # Steam cost: sum of heat utilities
            if hasattr(mee, 'heat_utilities'):
                total_steam = sum(util.cost for util in mee.heat_utilities.values())
                breakdown['Steam'] = total_steam * self.system.operating_hours
            # Vacuum pump electricity
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
        """Total annual OPEX including base and MEE unit costs."""
        # base OPEX from labor, maintenance, utilities, etc.
        base_opex = (
            self.system.power_utility.cost * self.system.operating_hours +
            self.CAPEX * (self.property_insurance + self.maintenance) +
            self.labor_cost * (1 + self.labor_burden)
        )
        return base_opex + sum(self.MEE_OPEX_breakdown.values())

    @property
    def MPSP(self):
        """Minimum product selling price, inherit default TEA calculation."""
        return super().MPSP


def create_vfa_tea_mee(system, OSBL_units=None):
    """Instantiate the VFA TEA model for MEE-focused analysis."""
    return VFA_TEA_MEE(system, OSBL_units=OSBL_units)