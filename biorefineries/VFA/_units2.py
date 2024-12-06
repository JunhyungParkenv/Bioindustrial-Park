# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:49:35 2024

@author: Junhyung Park
"""

from biosteam.units.design_tools.geometry import cylinder_diameter_from_volume
from thermosteam import MultiStream
from biosteam import Unit
from biosteam.units import Flash, HXutility, Mixer, MixTank, Pump, \
    SolidsSeparator, StorageTank, LiquidsSplitSettler
from biorefineries.make_a_biorefinery.utils import CEPCI, baseline_feedflow, compute_extra_chemical, adjust_recycle
from biosteam.units.decorators import cost
from biosteam.units.design_tools import size_batch
import thermosteam as tmo
import biosteam as bst
import numpy as np

Rxn = tmo.reaction.Reaction
ParallelRxn = tmo.reaction.ParallelReaction
_gal2m3 = 0.003785
_gpm2m3hr = 0.227124
# _m3hr2gpm = 4.40287
_hp2kW = 0.7457
_Gcal2kJ = 4184e3
#%% 
# Pretreatment

#%%
# =============================================================================
# Conversion
# =============================================================================
# Fermentation
@cost('Reactor volume', 'Anaerobic Reactor', cost=500000, S=500, CE=567.3, n=0.6, BM=2.5)
class UASB(Unit):
    _N_ins = 1
    _N_outs = 2  # VFA, Biogas
    _ins_size_is_fixed = False

    #: Fermentation temperature (K)
    T_fermentation = 40 + 273.15

    #: Operation time (hr, 40 days)
    tau_operation = 40 * 24

    #: Hydraulic Retention Time (HRT, hr, 20 days)
    tau_hrt = 20 * 24

    #: Unload and clean up time (hr)
    tau_0 = 4

    #: Working volume fraction (filled tank to total tank volume)
    V_wf = 0.95

    _units = {
        'Flow rate': 'm3/hr',
        'Reactor volume': 'm3',
        'Reactor duty': 'kJ/hr',
    }

    def __init__(self, ID='', ins=None, outs=(), P=101325):
        Unit.__init__(self, ID, ins, outs)
        self.P = P
        self.cofermentation_rxns = ParallelRxn([
            # Reaction definition                                          Reactant    Conversion
            Rxn('Glucose -> 0.0519 LacticAcid + 0.01 ValericAcid + 0.0632 ButyricAcid + 0.0119 PropionicAcid + 0.141 AceticAcid', 
                'Glucose', 
                0.9999),
        ])

    def _run(self):
        """Run the reaction and separate effluents."""
        feed = self.ins[0]
        vfa, biogas = self.outs

        # Copy input stream to effluent
        effluent = feed.copy()

        # Run the cofermentation reactions
        self.cofermentation_rxns(effluent.mol)

        # Assign effluent to the volatile fatty acids (VFAs) output
        vfa.copy_like(effluent)

        # Assume biogas only contains CO2 from reaction
        biogas.copy_flow(effluent, ('CO2',))
        vfa.imol['CO2'] = 0  # Remove CO2 from VFAs stream

        # Set temperature and pressure
        vfa.T = biogas.T = self.T_fermentation
        vfa.P = biogas.P = self.P

    def _design(self):
        """Design the reactor size and utility requirements."""
        feed = self.ins[0]
        vfa = self.outs[0]
        v_0 = feed.F_vol  # Total volumetric flow rate

        Design = self.design_results

        # HRT determines the reactor liquid volume
        V_liq = v_0 * self.tau_hrt

        # Total reactor volume based on operation time and working volume fraction
        V_tot = V_liq / self.V_wf

        Design['Flow rate'] = v_0
        Design['Reactor volume'] = V_tot

        # Reactor duty (energy requirement)
        self.add_heat_utility(vfa.Hnet - feed.Hnet, vfa.T)
#%%
# =============================================================================
# Separation
# =============================================================================
# ED
F = 96485.3

@cost('Membrane area', 'CEM', cost=100, S=1, CE=567.3, n=1, BM=2)
@cost('Membrane area', 'NF', cost=30, S=1, CE=567.3, n=1, BM=1.5)
@cost('Membrane area', 'Current Collector', cost=20, S=1, CE=567.3, n=1, BM=1.2)
@cost('Membrane area', 'Coating Solution', cost=0.057282, S=1, CE=567.3, n=1, BM=1.1)
@cost('Membrane area', 'Frames', cost=2, S=1, CE=567.3, n=1, BM=1.1)
@cost('Membrane area', 'Power supply', cost=20, S=1, CE=567.3, n=1, BM=1.3)
class ED(bst.Unit):
    _N_ins = 2
    _N_outs = 2

    def __init__(self, ID='', ins=None, outs=None, thermo=None, CE_dict=None, I=0.020092, 
                 A_m=None, R=39.75, z_T=1.0, t=24*3600, target_ratio=0.8):
        super().__init__(ID, ins, outs, thermo=thermo)
        # Exp : Assume C2 = 0.164472, C3 = 0.082236, C4 = 0.059, Assume C5 = 0.063118, C6 = 0.044
        self.CE_dict = CE_dict or {
            'AceticAcid': 0.164472, 'PropionicAcid': 0.082236, 'ButyricAcid': 0.059,
            'ValericAcid': 0.063118, 'LacticAcid': 0.082236
        }
        self.I = I           # Total current [A]
        self.A_m = A_m if A_m is not None else 1.0  # Default membrane area [m²] if not provided
        self.R = R           # System resistance [Ohm]
        self.z_T = z_T       # Charge number
        self.t = t           # Time in seconds for target concentration
        self.target_ratio = target_ratio  # Ratio of initial concentration to be reached
        
    def calculate_flux(self):
        # 각 이온의 플럭스를 전류에 따라 계산 (LacticAcid 제외)
        J_T_dict = {}
        for ion, CE in self.CE_dict.items():
            if ion != 'LacticAcid':  # LacticAcid는 플럭스 계산에서 제외
                J_T_dict[ion] = (CE * self.I) / (self.z_T * F * self.A_m)
        return J_T_dict

    def calculate_membrane_area(self, total_moles_to_transfer, total_flux):
        # 필요한 막 면적 계산
        A_m = total_moles_to_transfer / (total_flux * self.t)
        return A_m

    def _run(self):
        inf_dc, inf_ac = self.ins
        eff_dc, eff_ac = self.outs

        # 희석 compartment의 유량
        Q_dc = inf_dc.F_vol

        # 초기 총 VFA 양 (LacticAcid 제외)
        total_initial_vfa = sum(inf_dc.imol[ion] * 1e3 for ion in self.CE_dict if ion != 'LacticAcid')

        # 이동시킬 총 VFA 양 (80% 목표)
        total_vfa_to_transfer = total_initial_vfa * self.target_ratio

        # 각 이온의 플럭스 계산
        J_T_dict = self.calculate_flux()

        # 총 플럭스 계산
        total_flux = sum(J_T_dict.values())

        # 필요한 막 면적 계산
        self.A_m = self.calculate_membrane_area(total_vfa_to_transfer, total_flux)
        print(f"Calculated membrane area: {self.A_m:.2f} m²")

        # 막 면적이 변경되었으므로 플럭스 재계산
        J_T_dict = self.calculate_flux()

        # 각 이온별 이동량 계산 및 업데이트
        total_transferred_vfa = 0  # 실제로 이동된 총 VFA 양
        for ion in self.CE_dict:
            # 각 이온의 이동량 계산
            n_transferred = J_T_dict.get(ion, 0) * self.A_m * self.t
            available_amount = inf_dc.imol[ion] * 1e3
            actual_transfer = min(n_transferred, available_amount)
            
            eff_ac.imol[ion] = (inf_ac.imol[ion] * 1e3 + actual_transfer) / 1e3
            eff_dc.imol[ion] = (inf_dc.imol[ion] * 1e3 - actual_transfer) / 1e3
            
            if ion != 'LacticAcid':  # LacticAcid는 총 VFA 계산에 포함하지 않음
                total_transferred_vfa += actual_transfer

        # 물의 양 유지
        eff_dc.imol['Water'] = inf_dc.imol['Water']
        eff_ac.imol['Water'] = inf_ac.imol['Water']
        
    _units = {
        'Membrane area': 'm^2',
        'Tank volume': 'm^3',
        'System resistance': 'Ohm',
        'System voltage': 'V',
        'Power consumption': 'W',
        'Total current': 'A',
    }
    
    def _design(self):
        D = self.design_results
        # Store membrane area, current, resistance, and power calculations
        D['Membrane area'] = self.A_m
        D['Total current'] = self.I
        D['System resistance'] = self.R
        D['System voltage'] = D['Total current'] * self.R
        D['Power consumption'] = D['System voltage'] * D['Total current']

    def _cost(self):
        D = self.design_results
        self.baseline_purchase_costs['CEM'] = 2 * 100 * D['Membrane area']
        self.baseline_purchase_costs['NF'] = 30 * self.design_results['Membrane area']
        self.baseline_purchase_costs['Current Collector'] = 20 * self.design_results['Membrane area']
        self.baseline_purchase_costs['Coating Solution'] = 0.057282 * self.design_results['Membrane area']
        self.baseline_purchase_costs['Frames'] = 2 * self.design_results['Membrane area']
        self.baseline_purchase_costs['Power supply'] = 20 * D['Membrane area']
        self.power_utility.consumption = D['Power consumption'] / 1000  # Convert to kW
#%%
# =============================================================================
# Wastewater treatment
# =============================================================================

# Total cost of wastewater treatment is combined into this placeholder
@cost(basis='Flow rate', ID='Wastewater system', units='kg/hr', 
      kW=7018.90125, S=393100, cost=50280080, CE=CEPCI[2010], n=0.6, BM=1)
class WastewaterSystemCost(Unit): pass

class AnaerobicDigestion(Unit):
    """	
    Anaerobic digestion system as modeled by Humbird 2011	
    	
    Parameters	
    ----------  	
    ins :    	
        [0] Wastewater	
        	
    outs :   	
        [0] Biogas        	
        [1] Treated water        	
        [2] Sludge	
        	
    digestion_rxns: 
        [ReactionSet] Anaerobic digestion reactions.  	
    sludge_split: 
        [Array] Split between wastewater and sludge	
    	
    """
    auxiliary_unit_names = ('heat_exchanger',)
    _N_ins = 1	
    _N_outs = 3
    
    def __init__(self, ID='', ins=None, outs=(), *, reactants, split=(), T=35+273.15):	
        Unit.__init__(self, ID, ins, outs)	
        self.reactants = reactants	
        self.isplit = isplit = self.thermo.chemicals.isplit(split, None)
        self.split = isplit.data
        self.multi_stream = MultiStream(None)
        self.T = T
        self.heat_exchanger = hx = HXutility(None, None, None, T=T) 
        self.heat_utilities = hx.heat_utilities
        chems = self.chemicals	
        	
        # Based on P49 in Humbird et al., 91% of organic components is destroyed,	
        # of which 86% is converted to biogas and 5% is converted to sludge,	
        # and the biogas is assumed to be 51% CH4 and 49% CO2 on a dry molar basis	
        biogas_MW = 0.51*chems.CH4.MW + 0.49*chems.CO2.MW	
        f_CH4 = 0.51 * 0.86/0.91/biogas_MW	
        f_CO2 = 0.49 * 0.86/0.91/biogas_MW	
        f_sludge = 0.05 * 1/0.91/chems.WWTsludge.MW	
        	
        def anaerobic_rxn(reactant):	
            MW = getattr(chems, reactant).MW	
            return Rxn(f'{1/MW}{reactant} -> {f_CH4}CH4 + {f_CO2}CO2 + {f_sludge}WWTsludge',	
                       reactant, 0.91)	
        self.digestion_rxns = ParallelRxn([anaerobic_rxn(i) for i in self.reactants])
                	
    def _run(self):	
        wastewater = self.ins[0]	
        biogas, treated_water, sludge = self.outs	
        T = self.T	

        sludge.copy_flow(wastewater)	
        self.digestion_rxns(sludge.mol)	
        self.multi_stream.copy_flow(sludge)	
        self.multi_stream.vle(P=101325, T=T)	
        biogas.mol = self.multi_stream.imol['g']	
        biogas.phase = 'g'	
        liquid_mol = self.multi_stream.imol['l']	
        treated_water.mol = liquid_mol * self.split	
        sludge.mol = liquid_mol - treated_water.mol	
        # biogas.receive_vent(treated_water, accumulate=True)	
        biogas.receive_vent(treated_water)
        biogas.T = treated_water.T = sludge.T = T
        
    def _design(self):
        wastewater = self.ins[0]
        # Calculate utility needs to keep digester temperature at 35°C,	
        # heat change during reaction is not tracked	
        H_at_35C = wastewater.thermo.mixture.H(mol=wastewater.mol, 	
                                               phase='l', T=self.T, P=101325)	
        duty = -(wastewater.H - H_at_35C)
        self.heat_exchanger.simulate_as_auxiliary_exchanger(duty, wastewater)
  
class AerobicDigestion(Unit):
    """
    Anaerobic digestion system as modeled by Humbird 2011
    
    Parameters
    ----------
    ins :  
        [0] Wastewater        
        [1] Air
        [2] Caustic, added to neutralize the nitric acid produced by 
            nitrifying bacteria duing nitrification process
        
    outs :    
        [0] Vent
        [1] Treated wastewater
        
    digestion_rxns : 
        [ReactionSet] Anaerobic digestion reactions
    
    """
    
    _N_ins = 3
    _N_outs = 2
    # 4350, 4379, 356069, 2252, 2151522, and 109089 are water flows from 
    # streams 622, 630, 611, 632, 621, and 616  in Humbird et al.
    evaporation = 4350/(4379+356069+2252+2151522+109089)
    
    def __init__(self, ID='', ins=None, outs=(), *, reactants, ratio=0):
        Unit.__init__(self, ID, ins, outs)
        self.reactants = reactants
        self.ratio = ratio
        chems = self.chemicals
        
        def growth(reactant):
            f = chems.WWTsludge.MW / getattr(chems, reactant).MW 
            return Rxn(f"{f}{reactant} -> WWTsludge", reactant, 1.)
        
        # Reactions from auto-populated combustion reactions.
        # Based on P49 in Humbird et al, 96% of remaining soluble organic matter 
        # is removed after aerobic digestion, of which 74% is converted to
        # water and CO2 and 22% to cell mass
        combustion_rxns = chems.get_combustion_reactions()
        
        self.digestion_rxns = ParallelRxn([i*0.74 + 0.22*growth(i.reactant)
                                           for i in combustion_rxns
                                           if (i.reactant in reactants)])
        self.digestion_rxns.X[:] = 0.96
        
        #                                      Reaction definition       Reactant Conversion
        self.neutralization_rxn = Rxn('H2SO4 + 2 NaOH -> Na2SO4 + 2 H2O', 'H2SO4', 0.95)
    
    def _run(self):
        influent, air, caustic = self.ins
        vent, effluent = self.outs
        ratio = self.ratio
        vent.phase = 'g'

        # 51061 and 168162 from stream 630 in Humbird et al.
        air.imass['O2'] = 51061 * ratio
        air.imass['N2'] = 168162 * ratio
        # 2252 from stream 632 in Humbird et al
        caustic.imass['NaOH'] = 2252 * ratio
        caustic.imol['NaOH'] += 2 * influent.imol['H2SO4'] / self.neutralization_rxn.X
        caustic.imass['H2O'] = caustic.imass['NaOH']
        effluent.copy_like(influent)
        effluent.mol += air.mol
        effluent.mol += caustic.mol
        self.neutralization_rxn(effluent.mol)
        self.digestion_rxns(effluent.mol)
        vent.copy_flow(effluent, ('CO2', 'O2', 'N2'), remove=True)
        vent.imol['Water'] = effluent.imol['Water'] * self.evaporation
        effluent.imol['Water'] -= vent.imol['Water']
        
        # Assume NaOH is completely consumed by H2SO4 and digestion products
        effluent.imol['NaOH'] = 0
#%% Simple unit operations

