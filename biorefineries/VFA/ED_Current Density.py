import thermosteam as tmo
import biosteam as bst
import numpy as np
import matplotlib.pyplot as plt
from chaospy import distributions as shape
# Define other chemicals
AminoAcid = tmo.Chemical('AminoAcid', search_ID='Glycine')
FattyAcid = tmo.Chemical('FattyAcid', search_ID='PalmiticAcid')
Glucose = tmo.Chemical('Glucose')
Water = tmo.Chemical('Water')
Ethanol = tmo.Chemical('Ethanol')
PropionicAcid = tmo.Chemical('PropionicAcid') # C3
ButyricAcid = tmo.Chemical('ButyricAcid') # C4
LacticAcid = tmo.Chemical('LacticAcid') # C3
AceticAcid = tmo.Chemical('AceticAcid') # C2
ValericAcid = tmo.Chemical('ValericAcid') # C5
HexanoicAcid = tmo.Chemical('HexanoicAcid', search_ID='CaproicAcid') # C6
NaCl = tmo.Chemical('NaCl')

# Add all chemicals to the Chemicals object and set the thermo
# chemicals = tmo.Chemicals([AminoAcid, FattyAcid, Water, Ethanol, Glucose,
#                            PropionicAcid, ButyricAcid, LacticAcid, AceticAcid, ValericAcid, NaCl])

# TS1 Effluent
chemicals = tmo.Chemicals([Water,
                           PropionicAcid, ButyricAcid, LacticAcid, AceticAcid, ValericAcid, HexanoicAcid, NaCl])
tmo.settings.set_thermo(chemicals)

# Define inf_dc stream with 80% of full flow rate for 1 MGD
inf_dc = bst.Stream('inf_dc', 
                    Water=8751.4 * 0.8,         # kmol/hr for 80% of 1 MGD flow rate
                    AceticAcid=2.48 * 0.8,      # kmol/hr
                    PropionicAcid=0.21 * 0.8,   # kmol/hr
                    ButyricAcid=1.11 * 0.8,     # kmol/hr
                    LacticAcid=4.09 * 0.8,      # kmol/hr
                    ValericAcid=0.18 * 0.8,     # kmol/hr
                    units='kmol/hr')

# Define inf_ac stream with 20% of full flow rate for 1 MGD
inf_ac = bst.Stream('inf_ac', 
                    Water=8751.4 * 0.2,         # kmol/hr for 20% of 1 MGD flow rate
                    AceticAcid=2.48 * 0.2, # kmol/hr
                    PropionicAcid=0.21 * 0.2,  # kmol/hr
                    ButyricAcid=1.11 * 0.2,    # kmol/hr
                    LacticAcid=4.09 * 0.2,     # kmol/hr
                    ValericAcid=0.18 * 0.2,    # kmol/hr
                    units='kmol/hr')

inf_dc.show(N=100)
inf_ac.show(N=100)

# Create effluent streams
eff_dc = bst.Stream('eff_dc')
eff_ac = bst.Stream('eff_ac')

F = 96485.3

class ED_vfa(bst.Unit):
    _N_ins = 2
    _N_outs = 2
# R=39.75, A=0.0016m2
    def __init__(self, ID='', ins=None, outs=None, thermo=None, CE_dict=None, j=5.058, 
                 A_m=None, R=0.0000222, z_T=1.0, t=24*3600, target_ratio=0.8):
        super().__init__(ID, ins, outs, thermo=thermo)
        self.CE_dict = CE_dict or {
            'AceticAcid': 0.164472, 'PropionicAcid': 0.082236, 'ButyricAcid': 0.059,
            'ValericAcid': 0.063118, 'LacticAcid': 0.082236
        }
        self.j = j           # Current density [A/m²]
        self.A_m = A_m if A_m is not None else 1.0  # Default membrane area [m²] if not provided
        self.R = R           # System resistance [Ohm]
        self.z_T = z_T       # Charge number
        self.t = t           # Time in hours for target concentration
        self.target_ratio = target_ratio  # Ratio of initial concentration to be reached

    def calculate_flux(self, I):
        # 각 이온의 플럭스를 전류에 따라 계산 (LacticAcid 제외)
        J_T_dict = {}
        for ion, CE in self.CE_dict.items():
            if ion != 'LacticAcid':  # LacticAcid는 플럭스 계산에서 제외
                J_T_dict[ion] = (CE * I) / (self.z_T * F * self.A_m)
        return J_T_dict

    def calculate_membrane_area(self, total_moles_to_transfer, total_flux):
        # 필요한 막 면적 계산
        A_m = total_moles_to_transfer / (total_flux * self.t)
        return A_m
    
    def calculate_tank_volumes(self, Q_dc, HRT, ratio_ac_to_dc=0.2/0.8):
        V_dc = Q_dc * HRT  # Volume = Flow rate × HRT
        V_ac = V_dc * ratio_ac_to_dc
        return {'V_dc': V_dc, 'V_ac': V_ac}

    def _run(self):
        inf_dc, inf_ac = self.ins
        eff_dc, eff_ac = self.outs

        Q_dc = inf_dc.F_vol * 1000  # Convert to L/hr assuming F_vol in m³/hr
        HRT = 24  # Hydraulic Retention Time in hours
        tank_volumes = self.calculate_tank_volumes(Q_dc, HRT)
        
        self.design_results['DC Tank Volume'] = tank_volumes['V_dc']
        self.design_results['AC Tank Volume'] = tank_volumes['V_ac']
        
        print(f"Calculated DC Tank Volume: {tank_volumes['V_dc']:.2f} L")
        print(f"Calculated AC Tank Volume: {tank_volumes['V_ac']:.2f} L")

        # 초기 총 VFA 양 (LacticAcid 제외)
        total_initial_vfa = sum(inf_dc.imol[ion] * 1e3 for ion in self.CE_dict if ion != 'LacticAcid')

        # 이동시킬 총 VFA 양 (80% 목표)
        total_vfa_to_transfer = total_initial_vfa * self.target_ratio

        # 전류 계산
        I = self.j * self.A_m

        # 각 이온의 플럭스 계산
        J_T_dict = self.calculate_flux(I)

        # 총 플럭스 계산
        total_flux = sum(J_T_dict.values())

        # 필요한 막 면적 계산
        self.A_m = self.calculate_membrane_area(total_vfa_to_transfer, total_flux)
        print(f"Calculated membrane area: {self.A_m:.2f} m²")

        # 막 면적이 변경되었으므로 전류 및 플럭스 재계산
        I = self.j * self.A_m
        J_T_dict = self.calculate_flux(I)

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
        'Membrane area': 'm^2',  # Units for membrane area
        'Tank volume': 'm^3',  # Units for tank volume
        'System resistance': 'Ohm',  # Units for system resistance
        'System voltage': 'V',  # Units for system voltage
        'Power consumption': 'W',  # Units for power consumption
        'Total current': 'A',  # Units for total current
    }
    
    def _design(self):
        D = self.design_results
        # Store membrane area, current, resistance, and power calculations
        D['Membrane area'] = self.A_m
        D['Total current'] = self.j * self.A_m
        D['System resistance'] = self.R
        D['System voltage'] = D['Total current'] * self.R
        D['Power consumption'] = D['System voltage'] * D['Total current']

    def _cost(self):
        D = self.design_results
        self.baseline_purchase_costs['CEM'] = 2 * 100 * D['Membrane area']  # $100 per m² of membrane
        self.baseline_purchase_costs['NF'] = 30 * self.design_results['Membrane area']
        self.baseline_purchase_costs['Current Collector'] = 20 * self.design_results['Membrane area']
        self.baseline_purchase_costs['Coating Solution'] = 0.057282 * self.design_results['Membrane area']
        self.baseline_purchase_costs['Frames'] = 2 * self.design_results['Membrane area']
        self.baseline_purchase_costs['Power supply'] = 20 * D['Membrane area']  # $20 per m²
        self.power_utility.consumption = D['Power consumption'] / 1000  # Convert to kW

#%% Create ED_vfa unit
ED1 = ED_vfa(
    ID='ED1',
    ins=[inf_dc, inf_ac],
    outs=[eff_dc, eff_ac],
    j=11.375, #11.375
    t=24*3600,  # Target concentration over 24 hours
    target_ratio=0.8  # Reach 80% of the initial concentration
)

# Run the simulation
ED1.simulate()
ED1.results()
ED1.show(N=100)
#%% Run extended simulation to 168 hours (7 days) to check steady state
extended_time = 168 * 3600  # Total simulation time of 168 hours in seconds
ED1.t = extended_time

# Define tank volumes based on flow rates and HRT
Q_dc = inf_dc.F_vol * 1000  # Convert flow rate to L/hr
Q_ac = inf_ac.F_vol * 1000  # Convert flow rate to L/hr
HRT = 24  # Retention time in hours
tank_volumes = ED1.calculate_tank_volumes(Q_dc, HRT)
V_dc = tank_volumes['V_dc'] / 1000  # Convert to m³ for dilute compartment
V_ac = tank_volumes['V_ac'] / 1000  # Convert to m³ for concentrate compartment

# Initialize lists for total VFA concentrations (excluding LacticAcid) over extended time
total_vfa_concentration_dc_ext = []
total_vfa_concentration_ac_ext = []
time_points_ext = range(0, int(ED1.t), 3600)  # Simulate every hour for efficiency

# Run simulation over extended time to observe steady state behavior
for time in time_points_ext:
    total_vfa_dc = 0
    total_vfa_ac = 0

    for ion in ED1.CE_dict:
        if ion != 'LacticAcid':  # Exclude LacticAcid from flux calculation
            # Calculate the ion transfer over one hour
            flux = ED1.calculate_flux(ED1.j * ED1.A_m).get(ion, 0)
            n_transferred = flux * ED1.A_m * 3600  # moles transferred in one hour
            available_amount = inf_dc.imol[ion] * 1e3  # Convert kmol to mol
            actual_transfer = min(n_transferred, available_amount)

            # Update molar amounts in dilute and concentrate compartments
            inf_dc.imol[ion] -= actual_transfer / 1e3  # Convert mol to kmol
            inf_ac.imol[ion] += actual_transfer / 1e3  # Convert mol to kmol

            # Accumulate total VFA moles for each compartment
            total_vfa_dc += inf_dc.imol[ion] * 1e3  # Convert kmol to mol
            total_vfa_ac += inf_ac.imol[ion] * 1e3  # Convert kmol to mol

    # Calculate total VFA concentrations in each compartment using tank volumes
    conc_vfa_dc = total_vfa_dc / V_dc  # mol/m³
    total_vfa_concentration_dc_ext.append(conc_vfa_dc * 1e3)  # Convert to mM

    conc_vfa_ac = total_vfa_ac / V_ac  # mol/m³
    total_vfa_concentration_ac_ext.append(conc_vfa_ac * 1e3)  # Convert to mM

# Convert time_points_ext to hours for plotting
time_points_hours_ext = [t / 3600 for t in time_points_ext]

# Plot the total VFA concentration changes over time in mM for extended simulation
plt.figure(figsize=(7, 5))
plt.plot(time_points_hours_ext, total_vfa_concentration_dc_ext, label="Total VFA (dc)", linestyle='--')
plt.plot(time_points_hours_ext, total_vfa_concentration_ac_ext, label="Total VFA (ac)", linestyle='-')

# Customize plot appearance
plt.xlabel('Time (hr)', fontsize=16, fontweight='bold')  # Set font size and weight
plt.ylabel('Total VFA Concentration (mM)', fontsize=16, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)  # Set tick label size
plt.show()
#%% Simulation for total VFA concentration changes in DC and AC
# Define hypothetical tank volumes for concentration calculations
Q_dc = inf_dc.F_vol * 1000  # Convert flow rate to L/hr
HRT = 24  # Retention time in hours
tank_volumes = ED1.calculate_tank_volumes(Q_dc, HRT)
V_dc = tank_volumes['V_dc'] / 1000  # Convert to m³ for consistency
V_ac = tank_volumes['V_ac'] / 1000  # Convert to m³ for consistency

# Initialize lists to store total VFA concentrations (excluding LacticAcid) over time
total_vfa_concentration_dc = []
total_vfa_concentration_ac = []
time_points = range(0, int(ED1.t), 3600)  # Simulate every hour for efficiency

# Simulate hourly updates for VFA concentrations
for time in time_points:
    total_vfa_dc = 0
    total_vfa_ac = 0

    for ion in ED1.CE_dict:
        if ion != 'LacticAcid':  # Exclude LacticAcid from flux calculation
            # Calculate the ion transfer over one hour
            flux = ED1.calculate_flux(ED1.j * ED1.A_m).get(ion, 0)
            n_transferred = flux * ED1.A_m * 3600  # moles transferred in one hour
            available_amount = inf_dc.imol[ion] * 1e3  # Convert kmol to mol
            actual_transfer = min(n_transferred, available_amount)

            # Update molar amounts in dilute and concentrate compartments
            inf_dc.imol[ion] -= actual_transfer / 1e3  # Convert mol to kmol
            inf_ac.imol[ion] += actual_transfer / 1e3  # Convert mol to kmol

            # Accumulate total VFA moles for each compartment
            total_vfa_dc += inf_dc.imol[ion] * 1e3  # Convert kmol to mol
            total_vfa_ac += inf_ac.imol[ion] * 1e3  # Convert kmol to mol

    # Calculate total VFA concentrations in each compartment (mM)
    conc_vfa_dc = (total_vfa_dc / V_dc) * 1e3  # Convert mol/m³ to mM
    total_vfa_concentration_dc.append(conc_vfa_dc)

    conc_vfa_ac = (total_vfa_ac / V_ac) * 1e3  # Convert mol/m³ to mM
    total_vfa_concentration_ac.append(conc_vfa_ac)

# Convert time_points to hours for plotting
time_points_hours = [t / 3600 for t in time_points]  # Convert seconds to hours

# Plot the total VFA concentration changes over time in mM
plt.figure(figsize=(7, 5))
plt.plot(time_points_hours, total_vfa_concentration_dc, label="Total VFA (dc)", linestyle='--')
plt.plot(time_points_hours, total_vfa_concentration_ac, label="Total VFA (ac)", linestyle='-')

# Customize plot appearance
plt.xlabel('Time (hr)', fontsize=16, fontweight='bold')
plt.ylabel('Total VFA Concentration (mM)', fontsize=16, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

#%%
# Updated function for plotting with new j values and mol/(m²·s) units on the y-axis
def plot_area_flux_relationship(unit, j_values):
    areas = []
    fluxes = []
    
    for j in j_values:
        # Update current density and recalculate related values
        unit.j = j
        I = unit.j * unit.A_m
        J_T_dict = unit.calculate_flux(I)
        
        # Sum the fluxes directly in mol/(m²·s) without converting to hr
        total_flux = sum(J_T_dict.values())  # Now in mol/(m²·s)
        
        # Calculate total VFA transfer based on target ratio (convert kmol to mol)
        total_vfa_to_transfer = sum(inf_dc.imol[ion] * 1e3 for ion in unit.CE_dict if ion != 'LacticAcid') * unit.target_ratio
        
        # Recalculate membrane area based on new total flux
        A_m = unit.calculate_membrane_area(total_vfa_to_transfer, total_flux)
        areas.append(A_m)
        fluxes.append(total_flux)  # Now in mol/(m²·s)

    # Plotting the relationship between j values, membrane area, and flux
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(j_values, areas, 'g-', label='Membrane Area (A)')
    ax2.plot(j_values, fluxes, 'b-', label='Total Flux (J)')

    ax1.set_xlabel('Current Density (j) [A/m²]', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Membrane Area (A) [m²]', color='g', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Total Flux (J) [mol/(m²·s)]', color='b', fontsize=16, fontweight='bold')
    
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    fig.tight_layout()
    plt.show()

# Define a new range of current density values
j_values = np.linspace(5, 25, 5)
plot_area_flux_relationship(ED1, j_values)