# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:49:35 2024

@author: Junhyung Park
"""

# from biosteam.units.design_tools.geometry import cylinder_diameter_from_volume
from thermosteam import MultiStream
from biosteam import Unit
from biosteam.units import Flash, HXutility, Mixer, MixTank, Pump, \
    SolidsSeparator, StorageTank, LiquidsSplitSettler
from biorefineries.make_a_biorefinery.utils import CEPCI, baseline_feedflow, compute_extra_chemical, adjust_recycle
from biosteam.units.decorators import cost
# from biosteam.units.design_tools import size_batch
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

        # print("Feed contents before reaction:")
        # print(feed.show())
        
        # Copy input stream to effluent
        effluent = feed.copy()
        
        # print("Effluent contents before reaction:")
        # print(effluent.show())
        
        # Run the cofermentation reactions
        self.cofermentation_rxns(effluent.mol)
        
        # print("Effluent contents after reaction:")
        # print(effluent.show())
        
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

# --- DC Tank (MixTank를 사용) ---
@cost('Volume', 'DC Tank', cost=1000, S=1, CE=567.3, n=0.7, BM=1.5)
class DC_Tank(MixTank):
    _units = {'Volume': 'm^3'}

    def __init__(self, ID='', ins=None, outs=(), thermo=None, tau=24):
        """
        Parameters
        ----------
        tau : float
            체류시간 (hr). 기본값은 24시간.
        ins : tuple of Streams
            첫 번째 inlet은 전체 유입(신선 feed와 recycle의 혼합 유량)으로 가정.
            두 번째 inlet은 (옵션) 추가 inlet; MixTank는 기본적으로 2개의 inlet을 요구함.
        """
        # MixTank는 기본적으로 _N_ins = 2로 정의되어 있음
        super().__init__(ID, ins, outs, thermo)
        self.tau = tau  # 체류시간 (hr)

    def _design(self):
        # 여기서는 첫 번째 inlet을 기준으로 총 유량을 계산합니다.
        feed = self.ins[0]
        Design = self.design_results
        Design['Volume'] = feed.F_vol * self.tau  # 체류시간과 유량 기반 부피 계산
        super()._design()

# --- AC Tank (MixTank를 사용) ---
@cost('Volume', 'AC Tank', cost=1000, S=1, CE=567.3, n=0.7, BM=1.5)
class AC_Tank(MixTank):
    _units = {'Volume': 'm^3'}

    def __init__(self, ID='', ins=None, outs=(), thermo=None, tau=6):
        """
        Parameters
        ----------
        tau : float
            체류시간 (hr). 기본값은 6시간.
        ins : tuple of Streams
            첫 번째 inlet은 전체 유입(신선 feed와 recycle의 혼합 유량)으로 가정.
            두 번째 inlet은 (옵션) 추가 inlet; MixTank는 기본적으로 2개의 inlet을 요구함.
        """
        super().__init__(ID, ins, outs, thermo)
        self.tau = tau  # 체류시간 (hr)

    def _design(self):
        feed = self.ins[0]
        Design = self.design_results
        Design['Volume'] = feed.F_vol * self.tau  # 체류시간과 유량 기반 부피 계산
        super()._design()
# --- Electrodialysis Unit (ED) ---  
# Constants  
F = 96485.3  # Faraday constant (C/mol)  

@cost('Membrane area', 'CEM', cost=100, S=1, CE=567.3, n=1, BM=2)
@cost('Membrane area', 'NF', cost=30, S=1, CE=567.3, n=1, BM=1.5)
@cost('Membrane area', 'Current Collector', cost=20, S=1, CE=567.3, n=1, BM=1.2)
@cost('Membrane area', 'Coating Solution', cost=0.057282, S=1, CE=567.3, n=1, BM=1.1)
@cost('Membrane area', 'Frames', cost=2, S=1, CE=567.3, n=1, BM=1.1)
class ED(bst.Unit):
    _N_ins = 2  # inf_dc, inf_ac
    _N_outs = 2  # eff_dc, eff_ac

    def __init__(self, ID='', ins=None, outs=(), thermo=None, CE_dict=None, j=5.058,
                 A_m=1.0, R=0.0000222, z_T=1.0, t=24*3600, target_concentration=80000):
        super().__init__(ID, ins, outs, thermo=thermo)
        self.CE_dict = CE_dict or {
            'AceticAcid': 0.164472, 'PropionicAcid': 0.082236, 'ButyricAcid': 0.059,
            'ValericAcid': 0.063118, 'LacticAcid': 0.082236, 'Water': 0.0
        }
        self.j = j  # 전류 밀도 (A/m²)
        self.A_m = A_m  # 멤브레인 면적 (m²), 시스템 모듈에서 조정
        self.R = R  # 시스템 저항 (Ohm)
        self.z_T = z_T  # 이온 전하수
        self.t = t  # 작동 시간 (초)
        self.target_concentration = target_concentration  # 목표 농도 (g/L)

    def calculate_flux(self, I):
        """이온별 플럭스 계산 (mol/m²/s)"""
        return {ion: (CE * I) / (self.z_T * F * self.A_m) for ion, CE in self.CE_dict.items()}

    def calculate_membrane_area(self, total_vfa_mol, total_flux, Q):
        """목표 농도를 기준으로 Membrane Area 계산"""
        if total_flux == 0:
            return self.A_m  # 기존 값 유지

        # 목표 농도에 맞춰야 하는 총 mol 수 계산
        target_mol_transfer = ((self.target_concentration / 60.05) / 1000) * Q # mol/hr
        new_A_m = (target_mol_transfer * self.t / 3600) / (total_flux * self.t) # m2

        return max(new_A_m, 0.5)  # 최소 1.0 m² 보장

    def _run(self):
        """ED 유닛 실행 (A_m은 시스템에서 조정)"""
        inf_dc, inf_ac = self.ins
        eff_dc, eff_ac = self.outs

        total_initial_vfa = sum(inf_dc.imol[ion] for ion in self.CE_dict if ion != 'LacticAcid')
        if total_initial_vfa == 0:
            eff_dc.copy_like(inf_dc)
            eff_ac.copy_like(inf_ac)
            return

        # 전류량 계산
        I = self.j * self.A_m
        J_T_dict = self.calculate_flux(I) # mol/m²/s

        for ion in self.CE_dict:
            # available_amount = inf_dc.imol[ion] # kmol/hr
            # n_transferred = J_T_dict[ion] * self.A_m * self.t # mol
            # actual_transfer = min(n_transferred, available_amount)
            available_amount = inf_dc.imol[ion] # kmol/hr
            n_transferred = J_T_dict[ion] / 1000 * self.A_m * 3600 # kmol/hr
            actual_transfer = min(n_transferred, available_amount)

            eff_ac.imol[ion] = inf_ac.imol[ion] + actual_transfer
            eff_dc.imol[ion] = inf_dc.imol[ion] - actual_transfer
            
        # eff_dc.imol['Water'] = inf_dc.imol['Water']
        # eff_ac.imol['Water'] = inf_ac.imol['Water']

        
    _units = {
        'Membrane area': 'm^2',
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