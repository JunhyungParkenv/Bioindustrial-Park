# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:00:23 2023

@author: sarangbhagwat
"""

import biosteam as bst
import thermosteam as tmo
import numpy as np
from math import exp as math_exp
from thermosteam import Stream
from biorefineries.TAL import units
from biorefineries.TAL.process_settings import price, CFs
from biorefineries.TAL.utils import find_split, splits_df
from biorefineries.TAL.chemicals_data import chemical_groups
from biosteam import SystemFactory
from flexsolve import IQ_interpolation
from scipy.interpolate import interp1d, interp2d

Rxn = tmo.reaction.Reaction
ParallelRxn = tmo.reaction.ParallelReaction

# %% Utils
R = 8.314
TAL_Hm = 30883.66976 # by Dannenfelser-Yalkowsky method
TAL_Tm = 458.15 # K
TAL_c = 6056.69421768496 # fitted parameter
TAL_c_by_R = TAL_c/R
TAL_Hm_by_R = TAL_Hm/R

def get_TAL_solubility_in_water(T): # mol TAL : mol (TAL+water)
    return math_exp(-(TAL_Hm_by_R) * (1/T - 1/TAL_Tm))/math_exp(TAL_c_by_R/T) 

def get_mol_TAL_dissolved(T, mol_water):
    TAL_x = get_TAL_solubility_in_water(T)
    return mol_water*TAL_x/(1-TAL_x)

# def get_T_to_dissolve_TAL_(mol_TAL, mol_water):
    
def get_TAL_solubility_in_water_gpL(T):
    return get_mol_TAL_dissolved(T, 1000./18.)*126.11004

def get_K(chem_ID, stream, phase_1, phase_2):
    return (stream[phase_1].imol[chem_ID]/stream[phase_1].F_mol)/max(1e-6, (stream[phase_2].imol[chem_ID]/stream[phase_2].F_mol))

def get_TAL_solublity_in_solvent_very_rough(T, solvent_ID='Hexanol', units='g/L'):
    temp_stream =\
        Stream('temp_stream_get_TAL_solublity_in_solvent_very_rough')
    mol_water = mol_solvent = 1000
    mol_TAL = get_mol_TAL_dissolved(T, mol_water)
    temp_stream.imol['Water'] = mol_water
    temp_stream.imol[solvent_ID] = mol_solvent
    temp_stream.imol['TAL'] = mol_TAL
    temp_stream.lle(T=T, P=temp_stream.P)
    # temp_stream.show(N=100)
    phase_1 = 'l' if temp_stream.imol['l', solvent_ID] > temp_stream.imol['L', solvent_ID] else 'L'
    phase_2 = 'L' if phase_1=='l' else 'l'
    K_TAL_in_extract = get_K('TAL', temp_stream, phase_1, phase_2)
    # print(K_TAL_in_extract)
    if units=='g/L':
        temp_stream_2 = Stream('temp_stream_2_get_TAL_solublity_in_solvent_very_rough')
        temp_stream_2.imol['TAL'] = K_TAL_in_extract*mol_TAL
        temp_stream_2.imol[solvent_ID] = mol_solvent
        return temp_stream_2.imass['TAL']/temp_stream_2.F_vol
    elif units=='mol/mol':
        return K_TAL_in_extract*mol_TAL/(mol_TAL+mol_solvent) # 

def get_TAL_solubility_in_hexanol():
    return 2.*0.0222/(2.*0.0222+0.951) # mol/mol; 2 * Marco's initial experimental solubility of 2.8 wt% at 21 C

def get_TAL_solubility_in_ethanol_ww():
    return 0.167682 # solubility of 157.425 g-TAL per L-ethanol

def get_TAL_decarboxylation_conversion(T=273.15+80.):
    return decarb_conv_interp(T)


## Adsorption recoveries and capacities based on experimental data from Singh group
ts = [0.166666667,	0.5,	1,	2]
Ts = [303.15, 323.15]
recoveries = [[0.791785714,	0.947,	0.960821429,	0.975035714],
[0.92402381,	0.956595238,	0.96297619,	0.9785]]
capacities = [[0.0739,	0.088386667,	0.089676667,	0.091003333],
[0.086242222,	0.089282222,	0.089877778,	0.091326667]]

# Interpolate adsorption capacities and recoveries against time and temperature
rec_interp = interp2d(ts, Ts, recoveries)
cap_interp = interp2d(ts, Ts, capacities)

# Decarboxylation conversion against temperature

Ts_decarb = 273.15 + np.array([30, 50, 80])
conversions_decarb = 0.01 * np.array([13.91875948, 18.4816961, 22.503871])
decarb_conv_interp = interp1d(Ts_decarb, conversions_decarb)
# 
# Acetone washing to remove impurities from KSA
def get_mass_acetone_needed_per_mass_KSA():
    return 2. #!!! temporary

#%% Fermentation

@SystemFactory(ID = 'TAL_fermentation_process',
               ins=[dict(ID='sugar_juice_or_slurry', Glucose=10, Water=100),
                    dict(ID='CSL', CSL=100),
               ],
                outs=[dict(ID='F301_top_product', Water=20),
                      dict(ID='fermentation_liquid_effluent', TAL=1, Water=100),
                      dict(ID='fermentation_vent', CO2=1),
                      dict(ID='seedtrain_vent', CO2=1),
                                ],
                                               )
def create_TAL_fermentation_process(ins, outs,):
    
    sugar_juice_or_slurry, CSL = ins
    F301_top_product, fermentation_liquid_effluent, fermentation_vent, seedtrain_vent = outs
      
    # =============================================================================
    # Fermentation streams
    # =============================================================================
    
    # Flow and price will be updated in EnzymeHydrolysateMixer

    
    # For diluting concentrated, inhibitor-reduced hydrolysate
    dilution_water = Stream('dilution_water', units='kg/hr')
    
    # =============================================================================
    # Fermentation units
    # =============================================================================
    F301 = bst.MultiEffectEvaporator('F301', ins=sugar_juice_or_slurry, outs=('F301_l', F301_top_product),
                                            P = (101325, 73581, 50892, 32777, 20000), V = 0.1)
    
    F301_P = bst.Pump('F301_P', ins=F301-0, P=101325., )
    
    # Cool hydrolysate down to fermentation temperature at 30°C
    H301 = bst.units.HXutility('H301', ins=F301_P-0, T=30+273.15,)
    
    M304 = bst.units.Mixer('M304', ins=(H301-0, dilution_water))
    M304.water_to_sugar_mol_ratio = 5.
    
    @M304.add_specification()
    def adjust_M304_water():
        M304_ins_1 = M304.ins[1]
        M304_ins_1.imol['Water'] = M304.water_to_sugar_mol_ratio * M304.ins[0].imol['Glucose', 'Xylose'].sum()
        M304._run()
    
    M304_H = bst.units.HXutility('M304_H', ins=M304-0, T=30+273.15, rigorous=False)
    
    # Mix pretreatment hydrolysate/enzyme mixture with fermentation seed
    S302 = bst.Splitter('S302', ins=M304_H-0,
                        outs = ('to_seedtrain', 'to_cofermentation'),
                        split = 0.07) # split = inoculum ratio
    
    # Cofermentation
    R302 = units.CoFermentation('R302', 
                                    ins=(S302-1, '', CSL),
                                    outs=(fermentation_liquid_effluent, fermentation_vent))
    @R302.add_specification()
    def include_seed_CSL_in_cofermentation(): # note: effluent always has 0 CSL
        R302._run()
        R302.ins[2].F_mass*=1./(1-S302.split[0])
    
    # ferm_ratio is the ratio of conversion relative to the fermenter
    R303 = units.SeedTrain('R303', ins=S302-0, outs=('seed', seedtrain_vent), ferm_ratio=0.9)
    
    T301 = units.SeedHoldTank('T301', ins=R303-0, outs=1-R302)


#%% Separation of TAL by exploiting the temperature-sensitivity of TAL solubility 

@SystemFactory(ID = 'TAL_separation_solubility_exploit_process',
               ins=[dict(ID='fermentation_broth', TAL=1, Water=100),
               ],
                outs=[dict(ID='decarboxylation_vent', CO2=20),
                      dict(ID='S401_solid', FermMicrobe=1, Water=1),
                      dict(ID='S402_liquid', PD=1, Water=99),
                      dict(ID='solid_TAL', TAL=1),
                                ],
                                               )
def create_TAL_separation_solubility_exploit_process(ins, outs,):
    
    fermentation_broth, = ins
    decarboxylation_vent, S401_solid, S402_liquid, solid_TAL = outs
    
    # =============================================================================
    # Separation streams
    # =============================================================================
     
    # None
    
    # =============================================================================
    # Separation units
    # =============================================================================

    # Fake unit to enable solid-liquid equilibrium for fermentation broth
    U401 = bst.Unit('U401', ins=fermentation_broth, outs=('fermentation_broth_first_sle'))
    @U401.add_specification()
    def U401_spec():
        U401_ins_0 = U401.ins[0]
        tot_TAL = U401_ins_0.imol['TAL']
        U401_outs_0 = U401.outs[0]
        U401_outs_0.copy_like(U401_ins_0)
        mol_TAL_dissolved = get_mol_TAL_dissolved(U401_outs_0.T, U401_outs_0.imol['Water'])
        # U401_outs_0.sle('TAL', U401_outs_0.T) #!!! TODO: use computationally cheaper way of changing from Stream to MultiStream
        U401_outs_0.phases = ('s', 'l')
        U401_outs_0.imol['l', 'TAL'] = min(mol_TAL_dissolved, tot_TAL)
        U401_outs_0.imol['s', 'TAL'] = tot_TAL - min(mol_TAL_dissolved, tot_TAL)
        
 
    # Change broth temperature to adjust TAL solubility
    H401 = bst.HXutility('H401', ins=U401-0, outs=('H401_0'), T=273.15+56.)
    @H401.add_specification()
    def H401_spec():
        H401_ins_0 = H401.ins[0]
        H401_ins_0_water=H401_ins_0.imol['Water']
        tot_TAL = H401_ins_0.imol['TAL']
        
        lb_T = H401_ins_0.T
        ub_T = 99.+273.15
        
        if tot_TAL>get_mol_TAL_dissolved(ub_T, H401_ins_0_water):
            H401.T=ub_T
        else:
            H401_spec_obj_fn = lambda T: get_mol_TAL_dissolved(T, H401_ins_0_water) - tot_TAL
            H401.T = IQ_interpolation(H401_spec_obj_fn, lb_T, ub_T, ytol=1e-3)
        
        H401._run()
        H401_outs_0 = H401.outs[0]
        mol_TAL_dissolved = get_mol_TAL_dissolved(H401_outs_0.T, H401_outs_0.imol['Water'])
        
        H401_outs_0.phases = ('l', 's')
        H401_outs_0.imol['l', 'TAL'] = min(mol_TAL_dissolved, tot_TAL)
        H401_outs_0.imol['s', 'TAL'] = max(0., round(tot_TAL - min(mol_TAL_dissolved, tot_TAL), 5))

        

    U402 = bst.FakeSplitter('U402', ins=H401-0, outs = ('thermally_decarboxylated_broth',decarboxylation_vent))
    U402.decarboxylation_rxns = ParallelRxn([
        Rxn('TAL + H2O -> PD + CO2', 'TAL',   0.25),
        ])
    
    @U402.add_specification()
    def U402_spec():
        U402_outs_0 = U402.outs[0]
        U402_outs_0.copy_like(U402.ins[0])
        U402_outs_0.phases = ('l', 's')
        U402.decarboxylation_rxns[0].X = get_TAL_decarboxylation_conversion(T=U402_outs_0.T)
        U402.decarboxylation_rxns[0](U402_outs_0['l'])
        U402.outs[1].imol['CO2'] = U402_outs_0.imol['l', 'CO2']
        U402.outs[1].phase = 'g'
        U402_outs_0.imol['l', 'CO2'] = 0.
    
        
    # # Remove solids from fermentation broth, modified from the pressure filter in Humbird et al.
    S401_index = [splits_df.index[0]] + splits_df.index[2:].to_list()
    S401_cell_mass_split = [splits_df['stream_571'][0]] + splits_df['stream_571'][2:].to_list()
    S401_filtrate_split = [splits_df['stream_535'][0]] + splits_df['stream_535'][2:].to_list()
    S401 = bst.units.SolidsCentrifuge('S401', ins=U402-0, outs=(S401_solid, 'S401_liquid_fraction'),
                                # moisture_content=0.50,
                                split=find_split(S401_index,
                                                  S401_cell_mass_split,
                                                  S401_filtrate_split,
                                                  chemical_groups), 
                                solids =\
                                    ['Xylan', 'Glucan', 'Lignin', 'FermMicrobe',\
                                      'Ash', 'Arabinan', 'Galactan', 'Mannan'])
    
    @S401.add_specification()
    def S401_TAL_split_spec():
        S401._run()
        S401_ins_0 = S401.ins[0]
        S401.outs[0].imol['s', 'TAL'] = S401_ins_0.imol['s', 'TAL']
        S401.outs[0].imol['l', 'TAL'] = 0.
        S401.outs[1].imol['l', 'TAL'] = S401_ins_0.imol['l', 'TAL']
        S401.outs[1].imol['s', 'TAL'] = 0.
        

    
    H402 = bst.HXutility('H402', ins=S401-1, outs=('H402_0'), T=273.15+1.)
    

    H402.TAL_solubility_multiplier = 1.
    @H402.add_specification()
    def H402_spec():
        H402._run()
        H402_ins_0 = H402.ins[0]
        tot_TAL = H402_ins_0.imol['TAL']
        H402_outs_0 = H402.outs[0]
        TAL_solubility = H402.TAL_solubility_multiplier * get_mol_TAL_dissolved(H402_outs_0.T, H402_outs_0.imol['Water'])
        H402_outs_0.phases = ('s', 'l')
        H402_outs_0.T = H402.T
        TAL_dissolved = min(TAL_solubility, tot_TAL)
        H402_outs_0.imol['l', 'TAL'] = TAL_dissolved
        H402_outs_0.imol['s', 'TAL'] = max(0, tot_TAL - TAL_dissolved)
        
    S402 = bst.units.SolidsCentrifuge('S402', ins=H402-0, outs=('S402_solid_fraction', S402_liquid),
                                # moisture_content=0.50,
                                split=find_split(S401_index,
                                                  S401_cell_mass_split,
                                                  S401_filtrate_split,
                                                  chemical_groups), solids =\
                                    ['Xylan', 'Glucan', 'Lignin', 'FermMicrobe',\
                                      'Ash', 'Arabinan', 'Galactan', 'Mannan'])
    @S402.add_specification()
    def S402_TAL_split_spec():
        S402_ins_0 = S402.ins[0]
        solid_TAL = float(S402_ins_0.imol['s', 'TAL'])
        S402_ins_0.imol['s', 'TAL'] = 0.
        S402._run()
        S402.outs[0].imol['s', 'TAL'] = solid_TAL
        S402.outs[1].imol['l', 'TAL'] = S402_ins_0.imol['l', 'TAL']
        S402_ins_0.imol['s', 'TAL'] = solid_TAL
        S402.outs[1].phases = ('l',)
        
    
    
    H403 = bst.HXutility('H403', ins=S402-0, outs=(solid_TAL), T=273.15+40.)

    
    
# %% Separation of TAL by adsorption on activated charcoal
@SystemFactory(ID = 'TAL_separation_adsorption_process',
               ins=[dict(ID='fermentation_broth', TAL=1, Water=100),
                    dict(ID='ethanol_desorption', Ethanol=1),
               ],
                outs=[dict(ID='cooled_TAL', TAL=1),
                      dict(ID='S401_solid', FermMicrobe=1, Water=1),
                      dict(ID='broth_post_adsorption', Water=99),
                      dict(ID='S403_cool_air', N2=1),
                      dict(ID='ethanol_for_other_downstream_uses', Ethanol=0),
                      dict(ID='S407_cool_air', N2=1),
                                ],
                                               )
def create_TAL_separation_adsorption_process(ins, outs,):
    
    fermentation_broth, Ethanol_desorption = ins
    cooled_TAL, S401_solid, broth_post_adsorption, S403_cool_air, ethanol_for_other_downstream_uses, S407_cool_air = outs

    # =============================================================================
    # Separation units
    # =============================================================================
    
    
    # # Remove solids from fermentation broth, modified from the pressure filter in Humbird et al.
    S401_index = [splits_df.index[0]] + splits_df.index[2:].to_list()
    S401_cell_mass_split = [splits_df['stream_571'][0]] + splits_df['stream_571'][2:].to_list()
    S401_filtrate_split = [splits_df['stream_535'][0]] + splits_df['stream_535'][2:].to_list()
    S401 = bst.units.SolidsCentrifuge('S401', ins=fermentation_broth, outs=(S401_solid, 'S401_liquid_fraction'),
                                # moisture_content=0.50,
                                split=find_split(S401_index,
                                                  S401_cell_mass_split,
                                                  S401_filtrate_split,
                                                  chemical_groups), 
                                solids =\
                                    ['Xylan', 'Glucan', 'Lignin', 'FermMicrobe',\
                                      'Ash', 'Arabinan', 'Galactan', 'Mannan'])
    
    H401 = bst.units.HXutility('H401', ins=S401-1, outs = ('broth_to_adsorbtion',), T=30. + 273.15)
    
    M401 = bst.Mixer('M401', ins=(Ethanol_desorption, '',), outs=('mixed_ethanol_for_desorption'))
    S402 = bst.FakeSplitter('S402', ins=M401-0, outs=('ethanol_to_A401', ethanol_for_other_downstream_uses))
    
    @M401.add_specification()
    def M401_spec():
        makeup_ethanol, recycled_ethanol = M401.ins
        # A401.run()
        M401._run()
        M401_outs_0 = M401.outs[0]
        M401_outs_0.imol['Ethanol'] = sum([i.imol['Ethanol'] for i in M401.ins])
        makeup_ethanol.imol['Ethanol'] = max(0., M401_outs_0.imol['Ethanol'] - recycled_ethanol.imol['Ethanol'])
        # S402.run()
        # M401._run()
    
    A401 = bst.AdsorptionColumnTSA(
        'A401', 
        # ins=[bst.Stream('feed', TAL=0.014, Water=1, units='kg/hr', T=30 + 273.15), 'ethanol'], 
        ins=[H401-0, S402-0, 'hot_air'],
        outs=[broth_post_adsorption, 'TAL_laden_ethanol', 'ethanol_laden_air'],
        superficial_velocity=7.2, # m/h; typical velocities are 4 to 14.4 m/h for liquids; Adsorption basics Alan Gabelman (2017) Adsorption basics Part 1. AICHE
        
        regeneration_velocity=14.4, # m/h; default value (updated in unit specification based on titer)
        
        cycle_time=2., # 1-2 hours required for thermal-swing-adsorption (TSA) for silica gels (add 1 hr for conservativeness); Seader, J. D., Separation Process Principles: Chemical and Biochemical Operations,” 3rd ed., Wiley, Hoboken, NJ (2011).
        
        # This is density of activated carbon packing, including voids.
        # So rho_adsorbent = (1 - epsilon) * rho where epsilon is the void fraction
        # and rho is the density of activated carbon with no voids.
        adsorbent='Activated carbon',
        rho_adsorbent=None, # Bulk density including void fraction; calculated based on void fraction and solid density
        rho_adsorbent_solid=700, # Solid density excluding void fraction (in kg/m3)  # Seader et al. Table 15.2
        
        void_fraction = 0.5, # v/v # Seader et al. Table 15.2
        adsorbent_capacity=0.091, # default value for unsaturated capacity (updated in unit specification); conservative heuristic from Seider et. al. (2017) Product and Process Design Principles. Wiley
        T_regeneration=30. + 273.15, 
        drying_time = 0.55, # h #!!! This is updated to 0.5 h after the first run
        T_air = 351.39 + 10., # K # TAL_chemicals.Ethanol.Tb + 10
        air_velocity = 2160, # m/h
        vessel_material='Stainless steel 316',
        vessel_type='Vertical',
        regeneration_fluid=dict(phase='l', Ethanol=1., units='kg/hr'),
        adsorbate_ID='TAL',  
        split=dict(TAL=0, Water=1, VitaminA=1., VitaminD2=1., FermMicrobe=1.),
        length_unused = 1.219, # m; 4 ft based on recommendation by Seader et al. (Separation Process Principles)
        target_recovery=0.99,
        wet_retention=0.5, # conservatively assume half a wash's worth of ethanol is retained in the column before dry air is passed through it
        K = 0.07795, # back-calculated for 1 wash from experimental measurements for 3 washes pooled together; 0.125 for 3-wash # constant desorption partition coefficient; calculated for 1 wash from experimental data for 3 washes pooled together
    )
    A401._default_equipment_lifetime['Activated carbon'] = 1.
    A401.adsorbent_cost['Activated carbon'] = price['Activated carbon'] # 41. $/ft^3
    
    @A401.add_specification
    def A401_spec(): # update recovery and capacity based on user-input adsorption time and temperature
        
        T = A401.ins[0].T
        t = A401.cycle_time
        capacity = cap_interp(t, T)
        A401.adsorbent_capacity = capacity[0]
        
        A401._run()
        
        M401.run()
        
        A401.ins[1].T = M401.outs[0].T
    
    
        
    F401 = bst.units.MultiEffectEvaporator('F401', ins=A401-1, outs=('F401_b', 'F401_t'), chemical='Ethanol',
                                            P = (101325, 73581, 50892, 32777, 20000), V = 0.7)
    
    F401.flash=False
    F401.TAL_solubility_in_ethanol_ww = get_TAL_solubility_in_ethanol_ww()
    
    @F401.add_bounded_numerical_specification(x0=1e-4, x1=1.-1e-4, ytol=1e-4)
    def F401_obj_fn(V):
        F401_b = F401.outs[0]
        # F401_ins_0 = F401.ins[0]
        # TAL_mass = F401_ins_0.imass['TAL']
        # F401_ins_0.imass['TAL'] = 0.
        F401.V = V
        F401._run()
        # F401_ins_0.imass['TAL']  =TAL_mass
        # F401_b.imass['TAL'] = TAL_mass
    
        return F401.TAL_solubility_in_ethanol_ww - F401_b.imass['TAL']/F401_b.F_mass
    
    P401 = bst.Pump('P401', ins=F401-1, P=101325.)

    F402 = bst.DrumDryer('F402', 
                         ins=(F401-0, 'F402_air', 'F402_natural_gas'),
                         outs=('dry_TAL', 'F402_hot_air', 'F402_emissions'),
                         moisture_content=0.05, 
                         split=0.,
                         moisture_ID='Ethanol')
    
    H407 = bst.units.HXutility(
        'H407', ins=F402-1, outs=('cooled_ethanol_laden_air'), 
        T=265.,
        rigorous=True
    )
    
    S407 = bst.units.FakeSplitter('S407', ins=H407-0, outs=(S407_cool_air, 'ethanol_recovered_from_air'))
    
    @S407.add_specification()
    def S407_spec():
        S407_ins_0 = S407.ins[0]
        S407_ins_0.phases=('l','g')
        S407.outs[0].mol[:] = S407_ins_0['g'].mol[:]
        S407.outs[1].mol[:] = S407_ins_0['l'].mol[:]
    

    H403 = bst.units.HXutility('H403', ins=F402-0, outs=(cooled_TAL), 
                               T=30.+273.15, rigorous=True)
    
    
    H402 = bst.units.HXutility(
        'H402', ins=A401-2, outs=('cooled_ethanol_laden_air'), 
        T=265.,
        rigorous=True
    )
    
    S403 = bst.units.FakeSplitter('S403', ins=H402-0, outs=(S403_cool_air, 'ethanol_recovered_from_air'))
    
    @S403.add_specification()
    def S403_spec():
        S403_ins_0 = S403.ins[0]
        S403.outs[0].mol[:] = S403_ins_0['g'].mol[:]
        S403.outs[1].mol[:] = S403_ins_0['l'].mol[:]
        
    M402 = bst.Mixer('M402', ins=(P401-0, S407-1, S403-1), outs=('recycled_ethanol',))
    M402-0-1-M401

    
#%% Unused: Recover TAL from waste stream

@SystemFactory(ID = 'additional_TAL_recovery_process',
               ins=[dict(ID='mixed_waste_stream', TAL=1, PD=1, Water=100),
               ],
                outs=[
                      dict(ID='recovered_TAL', TAL=1),
                      dict(ID='S409_cool_air', N2=1),
                      dict(ID='actual_waste_stream', Water=100),
                                ],
                                               )
def create_additional_TAL_recovery_process(ins, outs,):
    
    mixed_waste_stream, = ins
    recovered_TAL, S409_cool_air, actual_waste_stream = outs
    
    F405 = bst.DrumDryer('F405', 
                         ins=(mixed_waste_stream, 'F405_air', 'F405_natural_gas'),
                         outs=(recovered_TAL, 'F405_hot_air', 'F405_emissions'),
                         moisture_content=0.05, 
                         split=0.,
                         # moisture_ID='H2O',
                         )
    F405.isplit['PD'] = 1. # assume all PD evaporates along with water
    
    H409 = bst.units.HXutility(
        'H409', ins=F405-1, outs=('cooled_water_PD_laden_air'), 
        T=280.,
        rigorous=True
    )
    
    S409 = bst.units.FakeSplitter('S409', ins=H409-0, outs=(S409_cool_air, actual_waste_stream))
    
    @S409.add_specification()
    def S409_spec():
        S409_ins_0 = S409.ins[0]
        S409_ins_0.phases=('l','g')
        S409.outs[0].mol[:] = S409_ins_0['g'].mol[:]
        S409.outs[1].mol[:] = S409_ins_0['l'].mol[:]
    
    
#%% Upgrading TAL to sorbic acid

@SystemFactory(ID = 'TAL_separation_adsorption_process',
               ins=[dict(ID='solid_TAL', TAL=1, ),
                    dict(ID='ethanol_minimal', Ethanol=1),
                    dict(ID='H2_hydrogenation', H2=1),
                    dict(ID='KOH_hydrolysis', KOH=1),
                    dict(ID='acetone_purification', Acetone=1),
               ],
                outs=[dict(ID='KSA', KSA=1),
                      dict(ID='impurities_to_boiler', DHL=0.1),
                      dict(ID='S407_cool_air', N2=1),
                      dict(ID='S408_cool_air', N2=1),
                                ],
                                               )
def create_TAL_to_sorbic_acid_upgrading_process(ins, outs,):
    
    solid_TAL, ethanol_minimal, H2_hydrogenation, KOH_hydrolysis, acetone_purification = ins
    KSA, impurities_to_boiler, S407_cool_air, S408_cool_air = outs
    
    M405 = bst.Mixer('M405', ins=(solid_TAL, ethanol_minimal, ''),
                     outs=('TAL_in_ethanol'))
    
    @M405.add_specification()
    def M405_ethanol_spec():
        M405_TAL, M405_makeup_ethanol, M405_recycled_ethanol = M405.ins
        M405_mixed, = M405.outs
        mass_TAL = sum([i.imass['TAL'] for i in M405.ins])
        current_mass_ethanol = sum([i.imass['Ethanol'] for i in [M405_TAL, M405_recycled_ethanol]])
        required_mass_ethanol = mass_TAL/get_TAL_solubility_in_ethanol_ww() - mass_TAL
        M405_makeup_ethanol.imass['Ethanol'] = max(0., required_mass_ethanol-current_mass_ethanol)
        M405._run()
        
    R401 = units.HydrogenationReactor('R401', ins = (M405-0, '', H2_hydrogenation), 
                                      outs = 'HMTHP',
                                      vessel_material='Stainless steel 316',
                                      tau=12)
    
    R402 = units.DehydrationReactor('R402', ins = (R401-0, ''), 
                                               outs = 'PSA',
                                               vessel_material='Stainless steel 316',
                                               tau=12)
    
    R403 = units.RingOpeningHydrolysisReactor('R403', ins = (R402-0, '', KOH_hydrolysis), 
                                   outs = 'KSA_in_ethanol',
                                   vessel_material='Stainless steel 316',
                                   tau=12)
    
    F403 = bst.DrumDryer('F403', 
                         ins=(R403-0, 'F403_air', 'F403_natural_gas'),
                         outs=('dry_KSA', 'F403_hot_air', 'F403_emissions'),
                         moisture_content=0.05, 
                         split=0.,
                         moisture_ID='Ethanol')
    
    H407 = bst.units.HXutility(
        'H407', ins=F403-1, outs=('cooled_ethanol_laden_air'), 
        T=265.,
        rigorous=True
    )
    
    S407 = bst.units.FakeSplitter('S407', ins=H407-0, outs=(S407_cool_air, 'ethanol_recovered_from_air'))
    
    @S407.add_specification()
    def S407_spec():
        S407_ins_0 = S407.ins[0]
        S407.outs[0].mol[:] = S407_ins_0['g'].mol[:]
        S407.outs[1].mol[:] = S407_ins_0['l'].mol[:]
    
    S407-1-2-M405
    
    M406 = bst.Mixer('M406', ins=(F403-0, acetone_purification, ''),)
    
    @M406.add_specification()
    def M406_ethanol_spec():
        M406_TAL, M406_makeup_acetone, M406_recycled_acetone = M406.ins
        M406_mixed, = M406.outs
        mass_KSA = sum([i.imass['KSA'] for i in M406.ins])
        current_mass_acetone = sum([i.imass['Acetone'] for i in [M406_TAL, M406_recycled_acetone]])
        required_mass_acetone = mass_KSA * get_mass_acetone_needed_per_mass_KSA()
        M406_makeup_acetone.imass['Acetone'] = max(0., required_mass_acetone-current_mass_acetone)
        M406._run()
    
    
    S406 = bst.FakeSplitter('S406', ins=M406-0,
                        outs=(KSA, 'impurities_in_acetone'))
    
    @S406.add_specification()
    def S406_spec():
        S406_ins_0 = S406.ins[0]
        S406_outs_0, S406_outs_1 = S406.outs
        S406_outs_0.imol['KSA'] = S406_ins_0.imol['KSA']
        S406_outs_0.T = S406_ins_0.T
        S406_outs_1.copy_like(S406_ins_0)
        S406_outs_1.imol['KSA'] = 0.
        
    F404 = bst.DrumDryer('F404', 
                         ins=(S406-1, 'F404_air', 'F404_natural_gas'),
                         outs=(impurities_to_boiler, 'F404_hot_air', 'F404_emissions'),
                         moisture_content=0.05, 
                         split=0.,
                         moisture_ID='Acetone')
    
    H408 = bst.units.HXutility(
        'H408', ins=F404-1, outs=('cooled_acetone_laden_air'), 
        T=265.,
        rigorous=True
    )
    
    S408 = bst.units.FakeSplitter('S408', ins=H408-0, outs=(S408_cool_air, 'acetone_recovered_from_air'))
    
    @S408.add_specification()
    def S408_spec():
        S408_ins_0 = S408.ins[0]
        S408.outs[0].mol[:] = S408_ins_0['g'].mol[:]
        S408.outs[1].mol[:] = S408_ins_0['l'].mol[:]
    
    S408-1-2-M406