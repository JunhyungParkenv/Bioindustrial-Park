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

class ED(bst.Unit):
    _N_ins = 2
    _N_outs = 2
# R=39.75, A=0.0016m2
    def __init__(self, ID='', ins=None, outs=None, thermo=None, CE_dict=None, j=5.058, 
                 A_m=None, R=0.0000222, z_T=1.0, t=24*3600, dc_tau=24, target_ratio=0.8):
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
        self.dc_tau = dc_tau  # DC tank residence time [hr]
        
        # Initialize storage tanks
        self.dc_storage = bst.StorageTank(
            'DC_Tank',
            tau=dc_tau,
        )
        
        self.ac_storage = bst.StorageTank(
            'AC_Tank',
            tau=dc_tau / 4,
        )
        
    def calculate_flux(self, I):
        J_T_dict = {ion: (CE * I) / (self.z_T * F * self.A_m) for ion, CE in self.CE_dict.items()}
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

        # Calculate initial total VFA excluding LacticAcid
        total_initial_vfa = sum(inf_dc.imol[ion] * 1e3 for ion in self.CE_dict if ion != 'LacticAcid')

        # Calculate total VFA to transfer (80% of initial, excluding LacticAcid)
        total_vfa_to_transfer = total_initial_vfa * self.target_ratio

        # Calculate current and flux
        I = self.j * self.A_m
        J_T_dict = self.calculate_flux(I)
        total_flux = sum(J_T_dict.values())

        # Calculate required membrane area
        self.A_m = self.calculate_membrane_area(total_vfa_to_transfer, total_flux)

        # Recalculate current and flux with updated membrane area
        I = self.j * self.A_m
        J_T_dict = self.calculate_flux(I)
        
        # Update effluent streams
        for ion in self.CE_dict:
            n_transferred = J_T_dict[ion] * self.A_m * self.t
            available_amount = inf_dc.imol[ion] * 1e3
            actual_transfer = min(n_transferred, available_amount)
                
            eff_ac.imol[ion] = (inf_ac.imol[ion] * 1e3 + actual_transfer) / 1e3
            eff_dc.imol[ion] = (inf_dc.imol[ion] * 1e3 - actual_transfer) / 1e3

        eff_dc.imol['Water'] = inf_dc.imol['Water']
        eff_ac.imol['Water'] = inf_ac.imol['Water']

        # Simulate storage tanks
        self.dc_storage.ins[:] = [eff_dc]
        self.dc_storage.outs[:] = [eff_dc]
        self.ac_storage.ins[:] = [eff_ac]
        self.ac_storage.outs[:] = [eff_ac]

        self.dc_storage.simulate()
        self.ac_storage.simulate()
        
        # Update outs explicitly
        self.outs[0] = self.dc_storage.outs[0]
        self.outs[1] = self.ac_storage.outs[0]

        print(f"Calculated membrane area: {self.A_m:.2f} m²")
        print(f"DC Tank Volume: {self.dc_storage.design_results['Total volume']:.2f} m³")
        print(f"AC Tank Volume: {self.ac_storage.design_results['Total volume']:.2f} m³")
        
    _units = {
        'Membrane area': 'm^2',  # Units for membrane area
        'DC Tank Volume': 'm^3',  # Units for DC tank volume
        'AC Tank Volume': 'm^3',  # Units for AC tank volume
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
        self.baseline_purchase_costs['DC Tank'] = self.dc_storage.baseline_purchase_costs['Tank']
        self.baseline_purchase_costs['AC Tank'] = self.ac_storage.baseline_purchase_costs['Tank']
#%% Create ED_vfa unit
# Example usage
ED1 = ED(
    ID='ED_with_storage',
    ins=[inf_dc, inf_ac],
    outs=[eff_dc, eff_ac],
    j=11.375,  # Current density
    t=24*3600,  # Time
    target_ratio=0.8,  # Target ratio
    dc_tau=24  # DC tank residence time [hr]
)

ED1.simulate()
ED1.results()
#%%
# Initialize j_values (current densities)
j_values = np.linspace(5, 25, 5)  # Current density values in A/m²

# Initialize AC tank volume list
ac_tank_volumes = []

# Iterate over current density values
for j in j_values:
    # Set the current density in the ED1 object
    ED1.j = j  # Update current density in ED1

    # Calculate total flux and total VFA to transfer
    I = ED1.j * ED1.A_m  # Total current (A)
    J_T_dict = ED1.calculate_flux(I)  # Flux for each ion
    total_flux = sum(J_T_dict.values())  # Total flux in mol/(m²·s)

    # Calculate total VFA to transfer (excluding LacticAcid)
    total_vfa_to_transfer = sum(inf_dc.imol[ion] * 1e3 for ion in ED1.CE_dict if ion != 'LacticAcid') * ED1.target_ratio

    # Calculate AC flow rate based on VFA transfer and flux
    Q_ac = (total_vfa_to_transfer / (total_flux * 3600)) * 1000  # Flow rate in L/hr

    # Calculate AC tank volume based on flow rate and retention time
    V_ac = Q_ac * (ED1.dc_tau / 4) / 1000  # Convert L to m³
    ac_tank_volumes.append(V_ac)

# Plot AC tank volume vs. current density
plt.figure(figsize=(8, 5))
plt.plot(j_values, ac_tank_volumes, marker='o', linestyle='-', color='b', label="AC Tank Volume")
plt.xlabel('Current Density (j) [A/m²]', fontsize=14, fontweight='bold')
plt.ylabel('AC Tank Volume (V_AC) [m³]', fontsize=14, fontweight='bold')
plt.title('AC Tank Volume vs. Current Density', fontsize=16, fontweight='bold')
plt.grid(True)
plt.legend(fontsize=12)
plt.show()

#%% v, HRT
# Define HRT values for analysis
hrt_values = np.linspace(1, 48, 12)  # HRT in hours

# Initialize lists for tank volumes
dc_tank_volumes = []
ac_tank_volumes = []

# Calculate tank volumes for each HRT
for hrt in hrt_values:
    Q_dc = inf_dc.F_vol * 1000  # DC flow rate in L/hr
    tank_volumes = ED1.calculate_tank_volumes(Q_dc, hrt)
    dc_tank_volumes.append(tank_volumes['V_dc'])
    ac_tank_volumes.append(tank_volumes['V_ac'])

# Plot DC and AC tank volumes vs. HRT
plt.figure(figsize=(8, 5))
plt.plot(hrt_values, dc_tank_volumes, label="DC Tank Volume", marker='o')
plt.plot(hrt_values, ac_tank_volumes, label="AC Tank Volume", marker='s')
plt.xlabel('Hydraulic Retention Time (HRT) [hr]', fontsize=14, fontweight='bold')
plt.ylabel('Tank Volume [m³]', fontsize=14, fontweight='bold')
plt.title('Tank Volume vs. HRT', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
#%% AC, A
# Define HRT values for analysis
hrt_values = np.linspace(1, 24, 12)  # HRT in hours, from 1 to 24 hours

# Initialize lists to store AC tank sizes and corresponding membrane areas
ac_tank_sizes = []
membrane_areas = []

# Ratio of AC to DC tank size
ratio_ac_to_dc = 0.2 / 0.8  # From previous code

# Calculate AC tank size and membrane area for each HRT
for hrt in hrt_values:
    # Calculate DC tank size based on HRT and flow rate
    Q_dc = inf_dc.F_vol * 1000  # Flow rate in L/hr
    V_dc = Q_dc * hrt / 1000  # DC Tank size in m³
    
    # Calculate AC tank size using the ratio
    V_ac = V_dc * ratio_ac_to_dc
    ac_tank_sizes.append(V_ac)
    
    # Calculate membrane area
    total_flux = sum(ED1.calculate_flux(ED1.j * ED1.A_m).values())  # Total flux in mol/(m²·s)
    total_vfa_to_transfer = sum(inf_dc.imol[ion] * 1e3 for ion in ED1.CE_dict if ion != 'LacticAcid') * ED1.target_ratio
    A_m = total_vfa_to_transfer / (total_flux * hrt * 3600)  # Membrane area in m²
    membrane_areas.append(A_m)

# Plot AC tank size vs. membrane area
plt.figure(figsize=(8, 5))
plt.plot(ac_tank_sizes, membrane_areas, marker='o', linestyle='-', label='Membrane Area vs. AC Tank Volume')
plt.xlabel('AC Tank Volume (V_ac) [m³]', fontsize=16, fontweight='bold')
plt.ylabel('Membrane Area (A) [m²]', fontsize=16, fontweight='bold')
plt.title('Relationship Between AC Tank Volume and Membrane Area', fontsize=16, fontweight='bold')
plt.grid(True)
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()
#%% DC, AC, A
# Define HRT values for analysis
hrt_values = np.linspace(1, 24, 12)  # HRT in hours, from 1 to 24 hours

# Initialize lists to store DC/AC tank sizes and corresponding membrane areas
dc_tank_sizes = []
ac_tank_sizes = []
membrane_areas = []

# Ratio of AC to DC tank size
ratio_ac_to_dc = 0.2 / 0.8  # From previous code

# Calculate tank sizes and membrane area for each HRT
for hrt in hrt_values:
    # Calculate DC tank size based on HRT and flow rate
    Q_dc = inf_dc.F_vol * 1000  # Flow rate in L/hr
    V_dc = Q_dc * hrt / 1000  # DC Tank size in m³
    dc_tank_sizes.append(V_dc)
    
    # Calculate AC tank size using the ratio
    V_ac = V_dc * ratio_ac_to_dc
    ac_tank_sizes.append(V_ac)
    
    # Calculate membrane area
    total_flux = sum(ED1.calculate_flux(ED1.j * ED1.A_m).values())  # Total flux in mol/(m²·s)
    total_vfa_to_transfer = sum(inf_dc.imol[ion] * 1e3 for ion in ED1.CE_dict if ion != 'LacticAcid') * ED1.target_ratio
    A_m = total_vfa_to_transfer / (total_flux * hrt * 3600)  # Membrane area in m²
    membrane_areas.append(A_m)

# Plot DC/AC tank sizes vs. membrane area
plt.figure(figsize=(10, 6))
plt.plot(dc_tank_sizes, membrane_areas, marker='o', linestyle='-', label='Membrane Area vs. DC Tank Volume')
plt.plot(ac_tank_sizes, membrane_areas, marker='x', linestyle='--', label='Membrane Area vs. AC Tank Volume')
plt.xlabel('Tank Size (V) [m³]', fontsize=16, fontweight='bold')
plt.ylabel('Membrane Area (A) [m²]', fontsize=16, fontweight='bold')
plt.title('Relationship Between Tank Volume and Membrane Area', fontsize=16, fontweight='bold')
plt.grid(True)
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()
#%% Run extended simulation to 168 hours (7 days) to check steady state
# Extend simulation to 168 hours (7 days) to check steady state
extended_time = 168 * 3600  # Total simulation time of 168 hours in seconds
ED1.t = extended_time

# Define tank volumes based on flow rates and HRT
Q_dc = inf_dc.F_vol * 1000  # Convert flow rate to L/hr
HRT = ED1.dc_tau  # Retention time in hours
V_dc = Q_dc * HRT / 1000  # Convert to m³ for dilute compartment
V_ac = V_dc * (0.2 / 0.8)  # Maintain 20% volume ratio for concentrate compartment

# Initialize lists for total VFA concentrations (excluding LacticAcid) over extended time
total_vfa_concentration_dc_ext = []
total_vfa_concentration_ac_ext = []
time_points_ext = range(0, int(extended_time), 3600)  # Simulate every hour for efficiency

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
plt.plot(time_points_hours_ext, total_vfa_concentration_dc_ext, label="Total VFA (dc)", linestyle='--', linewidth=2)
plt.plot(time_points_hours_ext, total_vfa_concentration_ac_ext, label="Total VFA (ac)", linestyle='-', linewidth=2)

# Customize plot appearance
plt.xlabel('Time (hr)', fontsize=16, fontweight='bold')  # Set font size and weight
plt.ylabel('Total VFA Concentration (mM)', fontsize=16, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)  # Set tick label size
plt.title('Total VFA Concentration Over Time (7 Days)', fontsize=16, fontweight='bold')
plt.show()
#%% Simulation for total VFA concentration changes in DC and AC
# Define hypothetical tank volumes for concentration calculations
Q_dc = inf_dc.F_vol * 1000  # Convert flow rate to L/hr
Q_ac = inf_ac.F_vol  # DC flow rate [m^3/hr]
HRT = ED1.dc_tau  # Retention time in hours
V_dc = Q_dc * HRT / 1000  # Convert to m³ for consistency
V_ac = V_dc / 4  # Convert to m³ for consistency

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
plt.title('Total VFA Concentration Over Time (1 Day)', fontsize=16, fontweight='bold')
plt.show()
#%%
# Create a System that includes the ED_vfa unit
system = bst.System('ED_System', path=(ED1,))

# Define product streams for TEA (e.g., eff_ac as the primary product)
products = [eff_ac]
    
# Define TEA object
tea = bst.TEA(
    system=system,
    IRR=0.1,  # Internal Rate of Return
    duration=(2024, 2044),  # Project duration (20 years)
    depreciation='MACRS7',  # Depreciation schedule
    operating_days=330,  # Operating days per year
    income_tax=0.21,  # Income tax rate
    lang_factor=3.0,  # Lang factor for capital cost estimation
    construction_schedule=(0.5, 0.5),  # Construction investment fractions
    startup_months=6,  # Startup duration in months
    startup_FOCfrac=0.5,  # Fraction of FOC during startup
    startup_VOCfrac=0.75,  # Fraction of VOC during startup
    startup_salesfrac=0.5,  # Fraction of sales during startup
    WC_over_FCI=0.05,  # Working capital as a fraction of FCI
    finance_interest=0.08,  # Financing interest rate
    finance_years=10,  # Loan period in years
    finance_fraction=0.6,  # Fraction of capital financed
)

# Run TEA calculations
fci = tea.FCI  # Fixed Capital Investment
tci = tea.TCI  # Total Capital Investment
aoc = tea.AOC  # Annual Operating Costs (excluding depreciation)
voc = tea.VOC  # Variable Operating Costs
foc = tea.FOC  # Fixed Operating Costs

# Print results
print(f"Fixed Capital Investment (FCI): {fci:.2f} USD")
print(f"Total Capital Investment (TCI): {tci:.2f} USD")
print(f"Fixed Operating Costs (FOC): {foc:.2f} USD/yr")
print(f"Variable Operating Costs (VOC): {voc:.2f} USD/yr")
print(f"Annual Operating Costs (AOC): {aoc:.2f} USD/yr")
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