#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:59:29 2024

@author: wenjun
"""

import numpy as np
import thermosteam as tmo
import biosteam as bst
from biosteam import Flowsheet as F 
from biosteam import main_flowsheet
from biosteam import units, Stream, SystemFactory
from biosteam.process_tools import UnitGroup
from biorefineries.SAF import _chemicals
from biorefineries.SAF import _units
from biorefineries.cellulosic import units
from biorefineries.SAF.utils import convert_ethanol_wt_2_mol
from biorefineries.SAF._process_settings import add_utility_agent, price


add_utility_agent()
F = bst.Flowsheet('SAF')
bst.main_flowsheet.set_flowsheet(F)

#%%
@SystemFactory(
    ID='preprocessing_sys',
    ins=[dict(ID='feedstock'),
         dict(ID='imbibition_water'),
         dict(ID='rvf_wash_water'),
         dict(ID='H3PO4'),
         dict(ID='lime'),
         dict(ID='polymer')],
    outs=[dict(ID='bagasse_to_ethanol'),
          dict(ID='bagasse_to_CHP'),
          dict(ID='filter_cake'),
          dict(ID='fiber_fines'),
          dict(ID='condensate'),
          dict(ID='concentrated_juice')])
def preprocessing_sys(ins,outs):
    feedstock,imbibition_water,rvf_wash_water,H3PO4,lime,polymer = ins
    bagasse_to_ethanol,bagasse_to_CHP,filter_cake,fiber_fines,condensate,concentrated_juice = outs
    
    H3PO4_storage = bst.StorageTank('H3PO4_storage', ins=H3PO4, outs='')
    
    U101 = bst.ConveyingBelt('U101', ins=feedstock)
    U102 = bst.MagneticSeparator('U102', ins=U101-0, outs=('A','B'))
    U103 = bst.Shredder('U103', ins=U102-0,outs='')
    U104 = bst.CrushingMill('U104', ins=(U103-0,''),
                            split=dict(Ash=0.92,
                                       Glucan=0.92,
                                       Xylan=0.92,
                                       Arabinan=0.92,
                                       Lignin=0.92,
                                       Sucrose=0.04,
                                       Glucose=0.04,
                                       Fructose=0.04,
                                       Extract=0.04,
                                       Solids=1), 
                            moisture_content=0.5)
    # Correct imbibition water
    @U101.add_specification(run=True)
    def correct_imbibition_water():
        feed = U101.ins[0]
        imbibition_water.imass['Water'] = 0.25 * feed.F_mass

    # Mix in water
    M101 = bst.Mixer('M101',ins=('',imbibition_water),outs=1-U104)

    # Screen out fibers
    S101 = bst.VibratingScreen('S101',ins=U104-1,outs=('', 0-M101),
                               split=dict(Ash=0.35,
                                          Glucan=0.35,
                                          Xylan=0.35,
                                          Arabinan=0.35,
                                          Lignin=0.35,
                                          Solids=0,
                                          Sucrose=0.88,
                                          Glucose=0.88,
                                          Fructose=0.88,
                                          Extract=0.88,
                                          Water=0.88))

    # Store juice
    T101 = bst.StorageTank('T101',ins=S101-0,outs='untreated_juice',
                           tau=4,vessel_material='Carbon steel')

    U105 = bst.ConveyingBelt('U105',ins=U104-0,outs='bagasse')

    # Split bagasse
    S102 = bst.Splitter('S102',ins=U105-0,outs=[bagasse_to_ethanol,bagasse_to_CHP],split=0.95)

    # Heat up juice #
    H101 = bst.HXutility('H101',ins=T101-0,T=343.15,rigorous=True)
            
    # Mix with acid #
    T102 = bst.MixTank('T102',ins=[H101-0,H3PO4_storage-0])
            
    # Pump solution #
    P101 = bst.Pump('P101',ins=T102-0)
            
    # Mix lime solution #
    T104 = bst.MixTank('T104',ins=(P101-0,lime),tau=0.1)
    P102 = bst.Pump('P102',ins=T104-0)
            
    # Mix recycle with rvf #
    M103 = bst.Mixer('M103',ins=(P102-0, ''))
            
    # Heat #
    H102 = bst.HXutility('H102',M103-0,T=105+273.15)
            
    # Mix in flocculant polymer #
    T105 = bst.MixTank('T105',ins=(H102-0,polymer),tau=0.1)
            
    # Clarify #
    C101 = bst.Clarifier('C101',ins=T105-0,outs=('clarified_juice',''),
                         split=dict(Ash=0,
                                    CaO=0,
                                    Glucan=0,
                                    Xylan=0,
                                    Arabinan=0,
                                    Lignin=0,
                                    Flocculant=0.522,
                                    H3PO4=0.522,
                                    Glucose=0.522,
                                    Sucrose=0.522,
                                    Fructose=0.522,
                                    Extract=0.522,
                                    Water=0.522))
    # Remove solids as filter cake #
    C102 = bst.RVF('C102',ins=(C101-1,rvf_wash_water),outs=(filter_cake,''),
                           moisture_content=0.80,
                           split=dict(Ash=0.85,
                                      CaO=0.85,
                                      Glucan=0.85,
                                      Xylan=0.85,
                                      Arabinan=0.85,
                                      Lignin=0.85,
                                      Glucose=0.01,
                                      Sucrose=0.01,
                                      Fructose=0.01,
                                      Extract=0.01))

    P103 = bst.Pump('P103',ins=C102-1,outs=1-M103)
            
    # Specifications dependent on energy cane flow rate
    @U104.add_specification(run=True) 
    def correct_chemical_flows():
        feed = U104.ins[0]
        lime.imass['CaO', 'Water'] = 0.001 * feed.F_mass * np.array([0.046, 0.954])
        H3PO4.imass['H3PO4', 'Water'] = 0.00025 * feed.F_mass
        
    # Specifications within P102
    @P102.add_specification(run=True)
    def correct_wash_water():
        solids = P102.ins[0].imol['Ash', 'CaO', 'Glucan',
                                  'Xylan','Arabinan', 'Lignin'].sum()
        rvf_wash_water.imol['Water'] = 0.0574 * solids
               
    # Juice screening process
    S103 = bst.VibratingScreen('S103',ins=C101-0,outs=('screened_juice',fiber_fines),
                               split=dict(Ash=0.998,
                                          CaO=0.998,
                                          H3PO4=1.0,
                                          Flocculant=0.0,
                                          Glucan=0.0,
                                          Xylan=0.0,
                                          Arabinan=0.0,
                                          Lignin=0.0,
                                          Glucose=0.998,
                                          Fructose=0.998,
                                          Sucrose=0.998,
                                          Extract=0.998,
                                          Water=0.998))
    S103.mesh_opening=2

    H103 = bst.HXutility('H103',ins=S103-0,T=25+273.15)   
       
    # Juice concentrating process 
    S104 = bst.Splitter('S104',ins=H103-0,split=0.9)
    E101 = bst.MultiEffectEvaporator('E101', ins=S104-0,outs=('solids',condensate),
                                     V=0.1, V_definition='First-effect',
                                     P=(101325, 73581, 50892, 32777))
    M104 = bst.Mixer('M104', ins=(E101-0,S104-1),outs='')
    H104 = bst.HXutility('H104', ins=M104-0, outs=concentrated_juice, T=32+273.15)
        
    E101.target_sugar_concentration=0.23
    @E101.add_bounded_numerical_specification(x0=0,x1=1,xtol=1e-5,ytol=1e-2)
    def sugar_concentration_at_fraction_evaporation(V):
        E101.V=V
        E101.run_until(M104,inclusive=True)
        Glucose_concentration = M104.outs[0].get_mass_fraction('Glucose')
        Sucrose_concentration = M104.outs[0].get_mass_fraction('Sucrose')
        Fructose_concentration = M104.outs[0].get_mass_fraction('Fructose')
        Extract_concentration = M104.outs[0].get_mass_fraction('Extract')
        sugar_concentration = Glucose_concentration + Sucrose_concentration + Fructose_concentration + Extract_concentration
        return E101.target_sugar_concentration - sugar_concentration





#%%

@SystemFactory(
    ID='pretreatment_sys',
    ins=[dict(ID='warm_process_water'),
         dict(ID='pretreatment_steam'),
         dict(ID='bagasse_to_ethanol')],
    outs=[dict(ID='pretreatment_to_WWT'),
          dict(ID='pretreated_bagasse')])
def pretreatment_sys(ins,outs):
    warm_process_water,pretreatment_steam,bagasse_to_ethanol = ins
    pretreatment_to_WWT,pretreated_bagasse = outs
    #csolids_loading = 0.305
    T_pretreatment_reactor = 130.+273.15
    
    P = pretreatment_steam.chemicals['H2O'].Psat(T_pretreatment_reactor + 25)
    
    M201 = bst.SteamMixer('M201',ins=(bagasse_to_ethanol,pretreatment_steam,warm_process_water),
                          P=P,solids_loading=0.305)
    R201 = _units.PretreatmentReactor('R201',ins=M201-0,T=T_pretreatment_reactor)
    P201 = units.BlowdownDischargePump('P201',ins=R201-1)
    F201 = units.PretreatmentFlash('F201', ins=P201-0,P=101325,Q=0)
    M202 = bst.Mixer('M202',ins=(R201-0,F201-0))
    units.WasteVaporCondenser('H201',ins=M202-0,outs=pretreatment_to_WWT,V=0)

    P202 = units.HydrolyzatePump('P202',ins=F201-1)

    U201 = bst.HammerMill('U201', ins=P202-0,outs=pretreated_bagasse)





#%%    
   
@SystemFactory(
    ID='fermentation_sys',
    ins=[dict(ID='pretreated_bagasse'),
         dict(ID='concentrated_juice'),
         dict(ID='enzyme_M301'),
         dict(ID='water_M301'),
         dict(ID='CSL'),
         dict(ID='DAP'),
         dict(ID='water_U301')],
    outs=[dict(ID='U301_vent'),
          dict(ID='U302_cell_mass'),
          dict(ID='U302_to_WWT'),
          dict(ID='D302_to_WWT'),
          dict(ID='ethanol')])
          
def fermentation_sys(ins,outs):
    pretreated_bagasse,concentrated_juice,enzyme_M301,water_M301,CSL,DAP,water_U301 = ins
    U301_vent,U302_cell_mass,U302_to_WWT,D302_to_WWT,ethanol = outs
    
    M301 = _units.EnzymeHydrolysateMixer('M301',ins=(pretreated_bagasse,enzyme_M301,water_M301),outs='',
                                         enzyme_loading=20,solids_loading=0.2)
    
    H301 = bst.HXutility('H301', ins=M301-0,T=48+273.15)
    
    DAP_storage = _units.DAPStorageTank('DAP_storage', ins=DAP, outs='')
    
    S300_DAP = bst.ReversedSplitter('S300_DAP', ins=DAP_storage-0, outs=['DAP_R301', 'DAP_R302'])
    
    CSL_storage = _units.CSLStorageTank('CSL_storage', ins=CSL, outs='')
    
    S300_CSL = bst.ReversedSplitter('S300_CSL', ins=CSL_storage-0, outs=['CSL_R301', 'CSL_R302'])
    
    R301 = _units.SaccharificationAndCoFermentation('R301',ins=(H301-0,'',S300_CSL-0,S300_DAP-0,concentrated_juice), 
                                                    outs=('R301_g','effluent','side_draw'))
    R302 = _units.SeedTrain('R302', ins=(R301-2,S300_CSL-1,S300_DAP-1,),
                           outs=('R302_g','seed'))
    T301 = _units.SeedHoldTank('T301', ins=R302-1,outs=1-R301)

    M302 = bst.Mixer('M302',ins=(R301-0,R302-0),outs='fermentation_vapor')
    
    @DAP_storage.add_specification(run=True)
    def update_DAP_storage_DAP():
        R301.run()
        R302.run()
        DAP.imass['DAP'] = S300_DAP.outs[0].F_mass + S300_DAP.outs[1].F_mass
        
    @CSL_storage.add_specification(run=True)
    def update_CSL_storage_DAP():
        R301.run()
        R302.run()
        CSL.imass['CSL'] = S300_CSL.outs[0].F_mass + S300_CSL.outs[1].F_mass
    
    U301 = bst.VentScrubber('U301', ins=(water_U301, M302-0), 
                            outs=(U301_vent, 'U301_recycled'),
                            gas=('CO2', 'NH3', 'O2'))

    @U301.add_specification(run=True)
    def update_U301_water():
        water_U301 = U301.ins[0]
        M302.run_until(U301,inclusive=True)
        water_U301.imass['Water']=26836/21759 * M302.F_mass_in

    M303 = bst.Mixer('M303',ins=(R301-1,U301-1),outs='')
    T302 = _units.BeerTank('T302', ins=M303-0,outs='')

    # Heat up crude beer by exchanging heat with stillage
    H302 = bst.HXprocess('H302', ins=(T302-0,''),outs='',
                         phase0='l',phase1='l',U=1.28)

    # Remove solids from fermentation broth, based on the pressure filter in ref [1]
    # Moisture content is 35% in ref [1] but 25% in ref [2], used 35% to be conservative
    U302 = _units.CellMassFilter('U302', ins=H302-1,outs=(U302_cell_mass,U302_to_WWT),
                                 moisture_content=0.35,split=0.99)

    # Beer column
    xbot = convert_ethanol_wt_2_mol(0.00001) 
    ytop = convert_ethanol_wt_2_mol(0.5)

    D301 = bst.BinaryDistillation('D301',ins=H302-0,outs=('',''),k=1.25,Rmin=0.6,
                                  P=101325,y_top=ytop,x_bot=xbot,
                                  LHK=('Ethanol','Water'),
                                  tray_material='Stainless steel 304',
                                  vessel_material='Stainless steel 304')
    D301.reboiler.U = 1.85
    D301_P = bst.Pump('D301_P', ins=D301-1, outs=1-H302)
    D301_P.BM = 3.1

    # Mix recycled ethanol
    # M304 = bst.Mixer('M304', ins=(D301-0, ''))

    ytop = convert_ethanol_wt_2_mol(0.915)
    D302 = bst.BinaryDistillation('D302', ins=D301-0,outs=('',''), k=1.25, Rmin=0.6,
                                  P=101325, y_top=ytop, x_bot=xbot,
                                  LHK=('Ethanol', 'Water'),
                                  tray_material='Stainless steel 304',
                                  vessel_material='Stainless steel 304',
                                  is_divided=True)
                                        
    D302.reboiler.U = 1.85
    D302_P = bst.Pump('D302_P', ins=D302-1, outs=D302_to_WWT)
    D302_P.BM = 3.1
    # Not use molecular sieve
    # D302_H = bst.HXutility('D302_H', ins=D302-0, outs='', T=115+283.15, V=1)

    # Molecular sieve, split based on streams 515 and 511 in ref [1]
    # split_ethanol = 1 - 21673/27022
    # split_water = 1 - 108/2164
    # S301 = bst.MolecularSieve('S301', ins=D302_H-0, outs=(1-M304, ''),
    #                           split=(split_ethanol, split_water),
    #                           order=('Ethanol', 'Water'))
                                    
    # Condense ethanol product
    S301_H = bst.HXutility('S301_H', ins=D302-0, outs='ethanol_to_storage',
                           V=0, T=20+273.15)
   
    ethanol_storage = bst.StorageTank('ethanol_storage', ins=S301_H-0, outs=ethanol,tau=7*24)
    
#%%

@SystemFactory(
    ID='upgrading_sys',
    ins=[dict(ID='NaOH'),
         dict(ID='Syndol_catalyst'),
         dict(ID='first_catalyst'),
         dict(ID='second_catalyst'),
         dict(ID='ethanol'),
         dict(ID='hydrogen'),
         dict(ID='Como_catalyst'),],
    outs=[dict(ID='F401_to_WWT'),
          dict(ID='D401_heavy_impurities'),
          dict(ID='D402_top_product'),
          dict(ID='CH4_C2H6'),
          dict(ID='gasoline'),
          dict(ID='jet_fuel'),
          dict(ID='diesel'),
          dict(ID='spent_catalyst_R401'),
          dict(ID='spent_catalyst_R402'),
          dict(ID='spent_catalyst_R403'),
          dict(ID='spent_catalyst_R404'),])
          
def upgrading_sys(ins,outs):
    NaOH,Syndol_catalyst,first_catalyst,second_catalyst,ethanol,hydrogen,Como_catalyst = ins
    F401_to_WWT,D401_heavy_impurities,D402_top_product,CH4_C2H6,gasoline,jet_fuel,diesel,spent_catalyst_R401,spent_catalyst_R402,spent_catalyst_R403,spent_catalyst_R404 = outs
    
    P401 = bst.Pump('P401',ins=ethanol,P=4.5*101325)
    M400 = bst.Mixer('M401', (P401-0,''))
    H400 = bst.HXutility('H400',ins=M400-0,V=1,rigorous=True)

    R401 = _units.AdiabaticFixedbedDehydrationReactor('R401', ins=(H400-0,Syndol_catalyst),outs=('',spent_catalyst_R401))

    # Depressurize to 1 bar before quenching
    V401 = bst.IsenthalpicValve('V401', ins=R401-0,P=101325)

    # Simulate as a quench tower
    H402 = bst.HXutility('H402', ins=V401-0,T=40+273.15,outs='crude_ethylene',rigorous=True)
    S401 = bst.PhaseSplitter('S401', ins=H402-0,outs=('vapor','liquid'))

    # Recover ethylene from water in S401-1 and S402-1
    M401 = bst.Mixer('M401',ins=(S401-1,''),rigorous=False)
    
    # Split flash setting based on eq.plot_vle_binary_phase_envelope(['Ethylene', 'Water'], P=101325) NO VACCUM (T cannot meet)
    F401 = bst.SplitFlash('F401', ins=M401-0, outs=('recovered_ethylene',F401_to_WWT), 
                          T=230, P=101325, split=dict(Ethylene=0.999,
                                                      Water=0.0001))
    # Only consider ethylene and water in recovered_ethylene, ignore impurities to simplify
    M402 = bst.Mixer('M402',ins=(S401-0,F401-0),outs='')

    # Reduce temperature to 15 C
    # H403 = bst.HXutility('H403', ins=M402-0,outs='',T=15+273.15)

    # Pressurize to 22 bar before purification;
    # Based on Bioethylene Production from Ethanol: A Review and Techno-economical Evaluation
    C401 = bst.MultistageCompressor('C401', ins=M402-0,n_stages=3,pr=2.8,vle=True)

    # Condense water
    S402 = bst.PhaseSplitter('S402', ins=C401-0,outs=('vapor','liquid'))


    # Ethylene recyle from water loop 
    S402-1-1-M401

    # Remove CO2
    U402 = _units.CausticTower('U402', ins=(S402-0, NaOH),P=22*101325)

    # Remove water and ethanol
    U403 = bst.MolecularSieve('U403', ins=U402-0,outs=('water_and_ethanol','dried_ethylene'),
                              split=dict(Water=1,
                                         Ethanol=1))
    U403.outs[0].vle(T=U403.outs[0].T,P=U403.outs[0].P)
    U403-0-1-M400
   
    # Reduce temperature before cryogenic distillation low temperature + high pressure
    # Temperature is based on Bioethylene Production from Ethanol: A Review and Techno-economical Evaluation
    H404 = bst.HXutility('H404',ins=U403-1,T=-25+273.15)

    # Ethylene column by cryogenic distillation, remove heavy keys (Propylene,
    # Butadiene, Ethane, Diethyl ether, and Acetaldehyde) 
    D401 = bst.BinaryDistillation('D401', ins=H404-0, 
                                  outs=('D401_top_product', D401_heavy_impurities),
                                  LHK=('Ethylene', 'Ethane'),
                                  P=22*101325, 
                                  Lr=0.9999, Hr=0.9999, 
                                  k=1.2)
    # Stripper column by cryogenic distillation, remove light keys (H2, CarbonMonoxide, CH4) 
    # M403 = bst.Mixer('M403',ins=(D401-0,''))
    D402 = bst.BinaryDistillation('D402', ins=D401-0, 
                                  outs=(D402_top_product, 'D402_pure_ethylene'),
                                  LHK=('CH4', 'Ethylene'), 
                                  P=22*101325,
                                  Lr=0.99, Hr=0.999,
                                  k=1.2)
    
    H405 = bst.HXutility('H405', ins=D402-1, T=10+273.15)

    # Ethylene purity is 100%, ignore impurity for simplification but considered in LCA
    S403 = bst.Splitter('S403', ins=H405-0, outs=('pure_ethylene',CH4_C2H6),
                        split=dict(Ethylene=1))

    # First oligomerization
    R402 = _units.Oligomerization1_Reactor('R402', ins=(S403-0,first_catalyst),outs=('',spent_catalyst_R402))

    # Second oligomerization
    R403 = _units.Oligomerization2_Reactor('R403',ins=(R402-0,'',second_catalyst),outs=('',spent_catalyst_R403))

    # Recycle light olefins to R403
    D403 = bst.BinaryDistillation('D403', ins=R403-0,outs=('light_olefins','heavy_olefins'),
                                  LHK=('C7H14','C8H16'),
                                  Lr=0.9999,
                                  Hr=0.9999,
                                  k=1.2)
    D403-0-1-R403

    # Hydrogenation
    P402 = bst.Pump('P402', ins=D403-1, outs='')
    R404 = _units.HydrogenationReactor('R404', ins=(P402-0,hydrogen,Como_catalyst),outs=(spent_catalyst_R404,''))
    
    @R404.add_specification(run=True)
    def correct_hydrogen_flow():
        feed=R404.ins[0]
        h2=R404.ins[1]
        h2.imol['H2']=feed.F_mol

    # Fractionation

    D404 = bst.BinaryDistillation('D404', ins=R404-1,outs=('C6_C8_distillate','more_than_C9_bottoms'),
                                  LHK=('C8H18','C9H20'),
                                  Lr=0.9999,
                                  Hr=0.9999,
                                  k=2,
                                  is_divided=False)

    D405 = bst.BinaryDistillation('D405', ins=D404-1,outs=('C9_C16_distillate','more_than_C16_bottoms'),
                                  LHK=('C16H34','C18H38'),
                                  Lr=0.999,
                                  Hr=0.999,
                                  k=1.2,
                                  is_divided=False)

    # Standard temperature for storing gasoline is around 15 C
    H406 = bst.HXutility('H406', ins=D404-0,outs='gasoline1',T=15+273.15,rigorous=True)

    # Standard temperature for storing jet fuel is around 15 C
    H407 = bst.HXutility('H407', ins=D405-0,outs='jet_fuel1',T=15+273.15,rigorous=True)

    # Standard temperature for storing diesel is around 20 C
    H408 = bst.HXutility('H408', ins=D405-1,outs='diesel1',T=20+273.15,rigorous=True)
    
    gasoline_storage = bst.StorageTank('gasoline_storage', ins=H406-0, outs=gasoline, tau=7*24, vessel_type='Floating roof', vessel_material='Carbon steel')
    
    jet_storage = bst.StorageTank('jet_storage', ins=H407-0, outs=jet_fuel, tau=7*24, vessel_type='Floating roof', vessel_material='Carbon steel')
    
    diesel_storage = bst.StorageTank('diesel_storage', ins=H408-0, outs=diesel, tau=7*24, vessel_type='Floating roof', vessel_material='Carbon steel')
    
    
    
#%%

# Complete energycane to SAF system
@SystemFactory(ID='SAF_sys')
def SAF_sys(ins,outs):
    sys_1 = preprocessing_sys(ins = (Stream(ID='feedstock',
                                            Water=0.6,
                                            Sucrose=0.077,
                                            Glucose=0.007,
                                            Fructose=0.006,
                                            Ash=0.029,
                                            Glucan=0.129,
                                            Xylan=0.07,
                                            Arabinan=0.008,
                                            Lignin=0.071,
                                            Extract=0.004,
                                            total_flow=2000000/24/0.4,
                                            units='kg/hr',
                                            price=price['feedstock']),
                                     Stream(ID='imbibition_water',
                                            Water=90000,
                                            units='kg/hr',
                                            T=338.15,
                                            price=price['water']),
                                     Stream(ID='rvf_wash_water',
                                            Water=1000,
                                            units='kg/hr',
                                            T=100+273.15,
                                            price=price['water']),
                                     Stream(ID='H3PO4',
                                            H3PO4=74.23,
                                            Water=13.1,
                                            units='kg/hr',
                                            price=price['H3PO4']),
                                     Stream(ID='lime',
                                            CaO=333,
                                            Water=2200,
                                            units='kg/hr', 
                                            price=price['lime']),
                                     Stream(ID='polymer',
                                            Flocculant=0.83, 
                                            units='kg/hr',
                                            price=price['flocculant'])))
                                     
    sys_2 = pretreatment_sys(ins = (Stream(ID='warm_process_water',
                                           T=368.15,
                                           Water=1,
                                           P=4.7*101325,
                                           price=price['water']),
                                    Stream(ID='pretreatment_steam',
                                           phase='g',
                                           T=268+273.15,
                                           P=13*101325,
                                           Water=24534+3490,
                                           units='kg/hr'),
                                    sys_1.outs[0]))
                                    
    sys_3 = fermentation_sys(ins = (sys_2.outs[1],
                                    sys_1.outs[-1],
                                    Stream(ID='enzyme_M301',
                                           units='kg/hr',
                                           price=price['enzyme']),
                                    Stream(ID='water_M301',
                                           units='kg/hr',
                                           price=price['water']),
                                    Stream(ID='CSL',
                                           CSL=1,
                                           units='kg/hr',
                                           price=price['CSL']),
                                    Stream(ID='DAP',
                                           DAP=1,
                                           units='kg/hr',
                                           price=price['DAP']),
                                    Stream(ID='water_U301',
                                           Water=1,
                                           units='kg/hr',
                                           price=price['water'])))
                                    
    sys_4 = upgrading_sys(ins = (Stream(ID='NaOH',
                                       NaOH=0.5,
                                       Water=0.5,
                                       units='kg/hr',
                                       price=price['NaOH']),
                                 Stream(ID='Syndol_catalyst',
                                        Ash=1,
                                        phase='s',
                                        units='kg/hr',
                                        price=price['Syndol catalyst']),
                                 Stream(ID='first_catalyst',
                                        Ash=1,
                                        phase='s',
                                        units='kg/hr',
                                        price=price['Ni-loaded aluminosilicate catalyst']),
                                 Stream(ID='second_catalyst',
                                        Ash=1,
                                        phase='s',
                                        units='kg/hr',
                                        price=price['Aluminosilicate catalyst']),
                                 sys_3.outs[-1],
                                 Stream(ID='hydrogen',
                                        phase='g',
                                        units='kg/hr',
                                        price=price['h2']),
                                 Stream('Como_catalyst',
                                        Ash=1,
                                        phase='s',
                                        units='kg/hr',
                                        price=price['Como catalyst'])),
                          outs = (Stream(ID='F401_to_WWT'),
                                  Stream(ID='D401_heavy_impurities'),
                                  Stream(ID='D402_top_product'),
                                  Stream(ID='CH4_C2H6'),
                                  Stream(ID='gasoline',
                                         price=price['gasoline']),
                                  Stream(ID='jet_fuel',
                                         price=price['jet fuel']),
                                  Stream(ID='diesel',
                                         price=price['diesel']),
                                  Stream(ID='spent_catalyst_R401',
                                         Ash=1),
                                  Stream(ID='spent_catalyst_R402',
                                         Ash=1),
                                  Stream(ID='spent_catalyst_R403',
                                         Ash=1),
                                  Stream(ID='spent_catalyst_R404',
                                         Ash=1)
                                  ))
                                  
    #==============================================================================
    #                      Area 500 WWT
    #==============================================================================
    M501 = bst.Mixer(ID='M501',
                     ins=(F.pretreatment_to_WWT,
                          F.U302_to_WWT,
                          F.D302_to_WWT,
                          F.F401_to_WWT))
    WWT = bst.create_conventional_wastewater_treatment_system(ID='WWT',
                                                              ins=M501.outs[0],
                                                              outs=('biogas_from_wastewater_treatment',
                                                                    'sludge_from_wastewater_treatment',
                                                                    'RO_treated_water_from_wastewater_treatment',
                                                                    'brine_from_wastewater_treatment'), # brine goes to waste
                                                              autopopulate=False,
                                                              NaOH_price=price['NaOH'])
    
    #==============================================================================
    #                      Area 600 Facilities
    #==============================================================================
    # Scale ADP, CIP, FWT by flow 
    feedstock = sys_1.ins[0]
    
    get_flow_tpd = lambda: (feedstock.F_mass-feedstock.imass['H2O'])*24/907.185

    # Streams
    natural_gas = Stream('natural_gas',price=price['natural gas'])
    lime_boiler = Stream('lime_boiler', price=price['lime'])
    boiler_chems = Stream('boiler_chems', price=price['boiler chems'])
    ash = Stream('ash', price=price['ash disposal'])
    cooling_tower_chems = Stream('cooling_tower_chems', price=price['cooling tower chems'])
    system_makeup_water = Stream('system_makeup_water', price=price['water'])

    plant_air_in = bst.Stream('plant_air_in',phase='g',units='kg/hr',
                              N2=0.79*1372608*get_flow_tpd()/2205,
                              O2=0.21*1372608*get_flow_tpd()/2205)
    CIP_chems_in = Stream('CIP_chems_in',Water=145*get_flow_tpd()/2205, units='kg/hr')
    fire_water_in = Stream('fire_water_in', Water=8021*get_flow_tpd()/2205, units='kg/hr')
    
    # Stream mixer to boiler turbogenerator for burning
    M901 = bst.Mixer('M901',
                     ins=(F.bagasse_to_CHP,
                          F.filter_cake,
                          F.fiber_fines,
                          F.U302_cell_mass,
                          F.D401_heavy_impurities,
                          F.sludge_from_wastewater_treatment), # WWT.outs[1]
                     outs='total_effluent_to_be_burned')  
    
    M902 = bst.Mixer('M902',
                    ins=(F.D402_top_product,
                         F.biogas_from_wastewater_treatment), # WWT.outs[0]
                    outs='total_gas_to_be_burned')

    BT = bst.BoilerTurbogenerator('BT',
                                  ins=(F.total_effluent_to_be_burned, # M901 outs[0]
                                       F.total_gas_to_be_burned, # M902 outs[0]
                                       'boiler_makeup_water',
                                       natural_gas,
                                       lime_boiler,
                                       boiler_chems),
                                  outs=('gas_emission',
                                        'boiler_blowdown_water',
                                         ash),
                                  satisfy_system_electricity_demand=True,
                                  natural_gas_price=0, # price separately set by the stream
                                  ash_disposal_price=0, # price separately set by the stream
                                  ) # price separately set by the stream
                                  
    BT.register_alias('CHP')

    # Chilled water package for cooling requirements
    CWP = bst.ChilledWaterPackage('CWP')
                                     
    CT = bst.CoolingTower('CT')
    CT.ins[-1].price = price['cooling tower chems']
                                                         
    # All water used in the system, here only consider water usage
    system_makeup_water_streams = (F.boiler_makeup_water,
                                   CT.ins[1]) # Second ins of CT (cooling_tower_makeup_water)
                             
    system_process_water_streams = (F.imbibition_water,
                                    F.rvf_wash_water,
                                    F.warm_process_water,
                                    F.water_M301,
                                    F.water_U301)
                             
    PWC = bst.ProcessWaterCenter('PWC',
                                  ins=(F.RO_treated_water_from_wastewater_treatment, # WWT.outs[2],
                                      '',
                                      F.condensate, # E101.outs[1],
                                      system_makeup_water),
                                  outs=('','process_water', 'discharged_water'),
                                  makeup_water_streams=system_makeup_water_streams,
                                  process_water_streams=system_process_water_streams)

    ADP = bst.AirDistributionPackage('ADP', ins=plant_air_in, outs='plant_air_out')

    CIP = bst.CIPpackage('CIP', ins=CIP_chems_in, outs='CIP_chems_out')

    FWT = _units.FireWaterTank('FWT', ins=fire_water_in, outs='fire_water_out')
    
    HXN = bst.HeatExchangerNetwork('HXN',cache_network=True)
    
    
    
    
    
#%% System setup and process groups

sys = SAF_sys()

BT_sys = bst.System('BT_sys', path=(F.BT,))

preprocessing = UnitGroup('Preprocessing_group', units = [i for i in sys.units if i.ID[1]=='1'])
                                          
pretreatment = UnitGroup('Pretreatment_group', units = [i for i in sys.units if i.ID[1]=='2'])
                                          
fermentation = UnitGroup('Fermentation_group', units = [i for i in sys.units if i.ID[1]=='3'])
                                          
upgrading = UnitGroup('Upgrading_group', units = [i for i in sys.units if i.ID[1]=='4'])
                                          
wastewater_treatment = UnitGroup('WWT_group', units = (F.WWT,))

heat_exchange_network = UnitGroup('HXN_group', units = (F.HXN,))

boiler_turbogenerator = UnitGroup('BT_group', units = (F.BT,)) 

cooling_tower = UnitGroup('CT_group', units = (F.CT,)) 

facilities_no_hu = UnitGroup('Facilities_no_hu_group', units = (F.CIP,)) 


process_groups = [preprocessing, pretreatment, fermentation, upgrading,
                  wastewater_treatment, 
                  heat_exchange_network, 
                  boiler_turbogenerator,
                  cooling_tower, facilities_no_hu]

process_groups_dict = {}
for i in range(len(process_groups)):
    group = process_groups[i]
    process_groups_dict[group.name] = group           
                 

raw_meterial_storage = UnitGroup('raw_meterial_storage', units=(F.H3PO4_storage, F.DAP_storage, F.CSL_storage))
product_storage = UnitGroup('product_storage', units=(F.ethanol_storage, F.gasoline_storage, F.jet_storage, F.diesel_storage))
