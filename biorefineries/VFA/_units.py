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

        print("Feed contents before reaction:")
        print(feed.show())
        
        # Copy input stream to effluent
        effluent = feed.copy()
        
        print("Effluent contents before reaction:")
        print(effluent.show())
        
        # Run the cofermentation reactions
        self.cofermentation_rxns(effluent.mol)
        
        print("Effluent contents after reaction:")
        print(effluent.show())
        
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
# Filter to separate fermentation broth into products liquid and solid
@cost(basis='Solids flow rate', ID='Feed tank', units='kg/hr',
      cost=174800, S=31815, CE=CEPCI[2010], n=0.7, BM=2.0)
@cost(basis='Solids flow rate', ID='Feed pump', units='kg/hr',
      kW=74.57, cost=18173, S=31815, CE=CEPCI[2010], n=0.8, BM=2.3)
@cost(basis='Pressing air flow rate', ID='Filter pressing compressor', units='kg/hr',
      kW=111.855, cost=75200, S=808, CE=CEPCI[2009], n=0.6, BM=1.6)
@cost(basis='Solids flow rate', ID='Pressing air compressor reciever', units='kg/hr',
      cost=8000, S=31815, CE=CEPCI[2010], n=0.7, BM=3.1)
@cost(basis='Drying air flow rate', ID='Filter drying compressor', units='kg/hr',
      kW=1043.98, cost=405000, S=12233, CE=CEPCI[2009], n=0.6, BM=1.6)
@cost(basis='Solids flow rate', ID='Dry air compressor reciever', units='kg/hr',
      cost=17000, S=31815, CE=CEPCI[2010], n=0.7, BM=3.1)
@cost(basis='Solids flow rate', ID='Pressure filter', units='kg/hr',
      cost=3294700, S=31815, CE=CEPCI[2010], n=0.8, BM=1.7)
@cost(basis='Solids flow rate', ID='Filtrate discharge pump', units='kg/hr',
      # Power not specified, based on filtrate tank discharge pump
      kW=55.9275, cost=13040, S=31815, CE=CEPCI[2010], n=0.8, BM=2.3)
@cost(basis='Solids flow rate', ID='Filtrate tank', units='kg/hr',
      cost=103000, S=31815, CE=CEPCI[2010], n=0.7, BM=2.0)
@cost(basis='Filtrate flow rate', ID='Flitrate tank agitator', units='kg/hr',
      kW=5.59275, cost=26000,  S=337439, CE=CEPCI[2009], n=0.5, BM=1.5)
@cost(basis='Solids flow rate', ID='Filtrate tank discharge pump', units='kg/hr',
      kW=55.9275, cost=13040, S=31815, CE=CEPCI[2010], n=0.8, BM=2.3)
@cost(basis='Solids flow rate', ID='Cell mass wet cake conveyor', units='kg/hr',
      kW=7.457, cost=70000, S=28630, CE=CEPCI[2009], n=0.8, BM=1.7)
@cost(basis='Solids flow rate', ID='Cell mass wet cake screw',  units='kg/hr',
      kW=11.1855, cost=20000, S=28630, CE=CEPCI[2009], n=0.8, BM=1.7)
@cost(basis='Solids flow rate', ID='Recycled water tank', units='kg/hr',
      cost=1520,  S=31815, CE=CEPCI[2010], n=0.7, BM=3.0)
@cost(basis='Solids flow rate', ID='Manifold flush pump', units='kg/hr',
      kW=74.57, cost=17057, S=31815, CE=CEPCI[2010], n=0.8, BM=2.3)
@cost(basis='Solids flow rate', ID='Cloth wash pump', units='kg/hr',
      kW=111.855,cost=29154, S=31815, CE=CEPCI[2010], n=0.8, BM=2.3)
class CellMassFilter(SolidsSeparator):
    _N_ins = 1
    _units= {'Solids flow rate': 'kg/hr',
             'Pressing air flow rate': 'kg/hr',
             'Drying air flow rate': 'kg/hr',
             'Filtrate flow rate': 'kg/hr'}

    def _design(self):
        Design = self.design_results
        # 809 is the scaling basis of equipment M-505,
        # 391501 from stream 508 in ref [1]
        Design['Pressing air flow rate'] = 809/391501 * self.ins[0].F_mass
        # 12105 and 391501 from streams 559 and 508 in ref [1]
        Design['Drying air flow rate'] = 12105/391501 * self.ins[0].F_mass
        Design['Solids flow rate'] = self.outs[0].F_mass
        Design['Filtrate flow rate'] = self.outs[1].F_mass

# MultiEffectEvaporator (MEE)

# Electrodialysis (ED)
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

    def __init__(self, ID='', ins=None, outs=None, thermo=None, CE_dict=None, j=5.058, 
                 A_m=None, R=0.0000222, z_T=1.0, t=24*3600, dc_tau=24, target_ratio=0.8):
        super().__init__(ID, ins, outs, thermo=thermo)
        self.CE_dict = CE_dict or {
            'AceticAcid': 0.164472, 'PropionicAcid': 0.082236, 'ButyricAcid': 0.059,
            'ValericAcid': 0.063118, 'LacticAcid': 0.082236
        }
        self.j = j
        self.A_m = A_m if A_m is not None else 1.0
        self.R = R
        self.z_T = z_T
        self.t = t
        self.target_ratio = target_ratio
        self.dc_tau = dc_tau

        self.dc_storage = bst.StorageTank('DC_Tank', tau=dc_tau)
        self.ac_storage = bst.StorageTank('AC_Tank', tau=dc_tau / 4)

    def calculate_flux(self, I):
        if self.A_m <= 0:
            raise ValueError(f"{self.ID}: Membrane area (A_m) must be greater than zero.")
        if self.z_T <= 0:
            raise ValueError(f"{self.ID}: Total charge number (z_T) must be greater than zero.")
        J_T_dict = {ion: (CE * I) / (self.z_T * F * self.A_m) for ion, CE in self.CE_dict.items()}
        return J_T_dict

    def calculate_membrane_area(self, total_moles_to_transfer, total_flux):
        if total_flux <= 0:
            raise ValueError(f"{self.ID}: Total flux must be greater than zero for membrane area calculation.")
        A_m = total_moles_to_transfer / (total_flux * self.t)
        return A_m
    
    def calculate_tank_volumes(self, Q_dc, HRT, ratio_ac_to_dc=0.2/0.8):
        V_dc = Q_dc * HRT
        V_ac = V_dc * ratio_ac_to_dc
        return {'V_dc': V_dc, 'V_ac': V_ac}

    def _run(self):
        print(f"Running ED unit with A_m={self.A_m}, target_ratio={self.target_ratio}, dc_tau={self.dc_tau}")
        inf_dc, inf_ac = self.ins
        eff_dc, eff_ac = self.outs
    
        # 입력 스트림 확인
        print("inf_dc contents:")
        print(inf_dc.show())
        print("inf_ac contents:")
        print(inf_ac.show())
    
        # 총 VFA 계산
        total_initial_vfa = sum(inf_dc.imol[ion] * 1e3 for ion in self.CE_dict if ion != 'LacticAcid')
    
        if total_initial_vfa <= 0:
            raise ValueError(f"{self.ID}: `inf_dc` contains no valid VFAs for transfer.")
    
        if self.target_ratio <= 0 or self.target_ratio > 1:
            raise ValueError(f"{self.ID}: Target ratio must be between 0 and 1.")
    
        total_vfa_to_transfer = total_initial_vfa * self.target_ratio
    
        print(f"Total initial VFA: {total_initial_vfa} mmol, Target transfer: {total_vfa_to_transfer} mmol")
    
        if total_vfa_to_transfer <= 0:
            raise ValueError(f"{self.ID}: Target ratio leads to invalid VFA transfer calculation.")
    
        # 전류 계산
        I = self.j * self.A_m
        if I <= 0:
            raise ValueError(f"{self.ID}: Current (I) must be greater than zero.")
    
        # Flux 계산
        J_T_dict = self.calculate_flux(I)
        total_flux = sum(J_T_dict.values())
        if total_flux <= 0:
            raise ValueError(f"{self.ID}: Total flux must be greater than zero.")
    
        # 멤브레인 면적 재계산
        self.A_m = self.calculate_membrane_area(total_vfa_to_transfer, total_flux)
        if self.A_m <= 0:
            raise ValueError(f"{self.ID}: Membrane area (A_m) is zero or negative after calculation.")
    
        print(f"Final A_m: {self.A_m}, total_flux: {total_flux}, I: {I}")

        # I = self.j * self.A_m
        # J_T_dict = self.calculate_flux(I)
        
        # 초기화
        eff_dc.empty()
        eff_ac.empty()
        
        for ion in self.CE_dict:
            n_transferred = J_T_dict[ion] * self.A_m * self.t
            available_amount = inf_dc.imol[ion] * 1e3
            actual_transfer = min(n_transferred, available_amount)
                
            eff_ac.imol[ion] = (inf_ac.imol[ion] * 1e3 + actual_transfer) / 1e3
            eff_dc.imol[ion] = (inf_dc.imol[ion] * 1e3 - actual_transfer) / 1e3

        eff_dc.imol['Water'] = inf_dc.imol['Water']
        eff_ac.imol['Water'] = inf_ac.imol['Water']
        
        # StorageTank 연결 수정
        self.dc_storage.empty()
        self.ac_storage.empty()
        
        # StorageTank 입력 및 출력 스트림 설정
        self.dc_storage.ins[:] = [eff_dc.copy()]
        # self.dc_storage.outs[:] = [eff_dc]
        self.ac_storage.ins[:] = [eff_ac.copy()]
        # self.ac_storage.outs[:] = [eff_ac]
        
        # self.dc_storage.ins[:] = [eff_dc]
        # self.ac_storage.ins[:] = [eff_ac]
        
        # self.dc_storage.ins[:] = [eff_dc]
        # self.dc_storage.outs[:] = [eff_dc]
        # self.ac_storage.ins[:] = [eff_ac]
        # self.ac_storage.outs[:] = [eff_ac]

        self.dc_storage.simulate()
        self.ac_storage.simulate()
        
        # 최종 출력 스트림 설정
        self.outs[0].copy_like(self.dc_storage.outs[0])
        self.outs[1].copy_like(self.ac_storage.outs[0])
        
        # self.outs[0] = self.dc_storage.outs[0]
        # self.outs[1] = self.ac_storage.outs[0]

    _units = {
        'Membrane area': 'm^2',
        'DC Tank Volume': 'm^3',
        'AC Tank Volume': 'm^3',
        'System resistance': 'Ohm',
        'System voltage': 'V',
        'Power consumption': 'W',
        'Total current': 'A',
    }
    
    def _design(self):
        D = self.design_results
        D['Membrane area'] = self.A_m
        D['Total current'] = self.j * self.A_m
        D['System resistance'] = self.R
        D['System voltage'] = D['Total current'] * self.R
        D['Power consumption'] = D['System voltage'] * D['Total current']
        
#%% Crystallization (BatchCrystallizer)
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

