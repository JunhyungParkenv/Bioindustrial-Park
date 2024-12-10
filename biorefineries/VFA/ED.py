import thermosteam as tmo
import biosteam as bst
import numpy as np
import matplotlib.pyplot as plt
from chaospy import distributions as shape
from SALib.sample import saltelli
from SALib.analyze import sobol
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

    def __init__(self, ID='', ins=None, outs=None, thermo=None, CE_dict=None, I=5.0, 
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
        self.total_cost = sum(self.baseline_purchase_costs.values())
#%% Create ED_vfa unit
ED1 = ED_vfa(
    ID='ED1',
    ins=[inf_dc, inf_ac],
    outs=[eff_dc, eff_ac],
    I=0.020092,  # Current [A]
    t=24*3600,  # Target concentration over 24 hours
    target_ratio=0.8  # Reach 80% of the initial concentration
)

# Run the simulation
ED1.simulate()
ED1.results()
ED1.show(N=100)
#%%
# Define system
system = bst.System('ED_system', path=(ED1,))

# 주요 메트릭 정의
metrics = [
    bst.Metric('Power Consumption', lambda: ED1.design_results['Power consumption'], 'W'),
]

model = bst.Model(system, metrics)

# Define parameter distributions
@model.parameter(name='Current [A]', units='A', distribution=shape.Triangle(0.001, 0.020092, 0.03))
def set_current(I):
    ED1.I = I

@model.parameter(name='System Resistance [Ohm]', units='Ohm', distribution=shape.Uniform(30, 50))
def set_resistance(R):
    ED1.R = R

@model.parameter(name='Target ratio', units='', distribution=shape.Uniform(0.1, 1.0))
def set_target_ratio(target_ratio):
    ED1.target_ratio = target_ratio

# Monte Carlo Analysis
N_samples = 100
rule = 'L'
np.random.seed(1234)
samples = model.sample(N_samples, rule)
model.load_samples(samples)

print("Running Monte Carlo analysis...")
model.evaluate(notify=10)

# Results
results = model.table
print(results)

# Plot results
results[('-', 'Total Cost')].hist(bins=20)
plt.title('Distribution of Total Cost')
plt.xlabel('Power Consumption (W)')
plt.ylabel('Frequency')
plt.show()
#%%
# Define system for ED process
system = bst.System('ED_system', path=(ED1,))

# Define metrics for analysis
metrics = [
    bst.Metric('Membrane Area', lambda: ED1.design_results['Membrane area'], 'm²'),
    bst.Metric('Power Consumption', lambda: ED1.design_results['Power consumption'], 'W'),
]

# Create Model object
model = bst.Model(system, metrics)

# Define parameter distributions
@model.parameter(name='Current [A]', units='A', distribution=shape.Triangle(0.01, 0.02, 0.03))
def set_current(I):
    ED1.I = I

@model.parameter(name='System Resistance [Ohm]', units='Ohm', distribution=shape.Uniform(30, 50))
def set_resistance(R):
    ED1.R = R

@model.parameter(name='Target Ratio', units='', distribution=shape.Uniform(0.1, 1.0))
def set_target_ratio(target_ratio):
    ED1.target_ratio = target_ratio

# Create separate distributions for each CE in CE_dict
@model.parameter(name='CE Acetic Acid', units='', distribution=shape.Uniform(0.1, 0.3))
def set_ce_acetic(C):
    ED1.CE_dict['AceticAcid'] = C

@model.parameter(name='CE Propionic Acid', units='', distribution=shape.Uniform(0.05, 0.25))
def set_ce_propionic(C):
    ED1.CE_dict['PropionicAcid'] = C

@model.parameter(name='CE Butyric Acid', units='', distribution=shape.Uniform(0.05, 0.2))
def set_ce_butyric(C):
    ED1.CE_dict['ButyricAcid'] = C

@model.parameter(name='CE Valeric Acid', units='', distribution=shape.Uniform(0.05, 0.2))
def set_ce_valeric(C):
    ED1.CE_dict['ValericAcid'] = C

# Monte Carlo simulation
N_samples = 100
rule = 'L'  # Latin Hypercube Sampling
np.random.seed(1234)  # For consistent results
samples = model.sample(N_samples, rule)
model.load_samples(samples)

# Evaluate the model
print("Evaluating Monte Carlo samples...")
model.evaluate(notify=10)

# Display Monte Carlo results
results = model.table
print("Monte Carlo Results:")
print(results)

# Plot distribution of membrane area
if ('-', 'Membrane Area') in results.columns:
    results[('-', 'Membrane Area')].hist(bins=20)
    plt.title('Distribution of Membrane Area')
    plt.xlabel('Membrane Area (m²)')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Column ('-', 'Membrane Area') not found in results.")

# Perform Spearman's rank-order correlation
df_rho, df_p = model.spearman_r()

# Display Spearman correlation results for Membrane Area
if ('-', 'Membrane Area') in df_rho.columns:
    print("Spearman Correlation for Membrane Area:")
    print(df_rho[('-', 'Membrane Area')])
else:
    print("Column ('-', 'Membrane Area') not found in Spearman correlation results.")

# Tornado plot for sensitivity analysis
if ('-', 'Membrane Area') in df_rho.columns:
    bst.plots.plot_spearman_1d(
        df_rho[('-', 'Membrane Area')],
        index=[param.describe() for param in model.parameters],
        name='Membrane Area Sensitivity'
    )
else:
    print("Unable to create Tornado Plot: Column ('-', 'Membrane Area') not found in df_rho.")

#%%
# 시스템 정의
system = bst.System('ED_system', path=(ED1,))  # ED1 유닛을 포함한 시스템 정의

# 모델 정의
metrics = [
    bst.Metric('Membrane Area', lambda: ED1.design_results['Membrane area'], 'm²')
]

model = bst.Model(system, metrics)  # 모델 초기화

# 분포 정의 (범위를 조정하여 안정성 확보)
current_distribution = shape.Triangle(0.005, 0.02, 0.025)  # Current 범위 좁힘
resistance_distribution = shape.Uniform(35, 45)            # Resistance 범위 좁힘
CE_distribution = shape.Uniform(0.1, 0.4)                  # CE 범위 좁힘
target_ratio_distribution = shape.Uniform(0.2, 0.8)        # Target Ratio 범위 조정

# 파라미터 정의
@model.parameter(name='Current [A]', units='A', distribution=current_distribution)
def set_current(I):
    ED1.I = I

@model.parameter(name='System resistance [Ohm]', units='Ohm', distribution=resistance_distribution)
def set_resistance(R):
    ED1.R = R

@model.parameter(name='Current Efficiency (CE)', units='', distribution=CE_distribution)
def set_CE(CE):
    for ion in ED1.CE_dict:
        ED1.CE_dict[ion] = CE

@model.parameter(name='Target Ratio', units='', distribution=target_ratio_distribution)
def set_target_ratio(target_ratio):
    ED1.target_ratio = target_ratio

# Monte Carlo 분석
N_samples = 100
rule = 'L'  # Latin Hypercube Sampling
np.random.seed(1234)
samples = model.sample(N_samples, rule)
model.load_samples(samples)

# Evaluate the model
print("Evaluating Monte Carlo samples...")
model.evaluate(notify=10)

# Monte Carlo 결과 출력
results = model.table
print("Monte Carlo Results:")
print(results)

# 결과 시각화: Membrane Area 분포
if ('-', 'Membrane area [m²]') in results.columns:
    results[('-', 'Membrane area [m²]')].hist(bins=20)
    plt.title('Distribution of Membrane Area')
    plt.xlabel('Membrane Area (m²)')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Column ('-', 'Membrane area [m²]') not found in results.")

# Spearman 상관 분석
df_rho, df_p = model.spearman_r()

# Spearman 상관 계수 출력
if ('-', 'Membrane area [m²]') in df_rho.columns:
    print("Spearman Correlation for Membrane Area:")
    print(df_rho[('-', 'Membrane area [m²]')])
else:
    print("Column ('-', 'Membrane area [m²]') not found in Spearman correlation results.")

# Tornado 플롯 생성
if ('-', 'Membrane area [m²]') in df_rho.columns:
    bst.plots.plot_spearman_1d(
        df_rho[('-', 'Membrane area [m²]')],
        index=[param.describe() for param in model.parameters],
        name='Membrane Area Sensitivity'
    )
else:
    print("Unable to create Tornado Plot: Column ('-', 'Membrane area [m²]') not found in df_rho.")
#%%
# 시스템 정의
system = bst.System('ED_system', path=(ED1,))

# Define the model for uncertainty analysis
metrics = [
    bst.Metric('Membrane Area', lambda: ED1.design_results['Membrane area'], 'm²')
]

model = bst.Model(system, metrics)

# Define parameter distributions
@model.parameter(name='Current [A]', units='A', distribution=shape.Triangle(0.018, 0.020092, 0.022))
def set_current(I):
    ED1.I = I

@model.parameter(name='System resistance [Ohm]', units='Ohm', distribution=shape.Uniform(35, 45))
def set_resistance(R):
    ED1.R = R

@model.parameter(name='Current Efficiency (CE)', units='', distribution=shape.Uniform(0.05, 0.2))
def set_CE(CE):
    for ion in ED1.CE_dict:
        ED1.CE_dict[ion] = CE

@model.parameter(name='Target Ratio', units='', distribution=shape.Uniform(0.7, 0.9))
def set_target_ratio(target_ratio):
    ED1.target_ratio = target_ratio

# Perform Monte Carlo simulation
N_samples = 100
samples = model.sample(N_samples, rule='L')  # Latin-Hypercube Sampling
model.load_samples(samples)

# Evaluate the model
model.evaluate(notify=10)

# Display results
results = model.table
print("Monte Carlo Results:")
print(results)

# Visualize the distribution of membrane area
results[('-', 'Membrane area [m²]')].hist(bins=20)
plt.title('Distribution of Membrane Area')
plt.xlabel('Membrane Area (m²)')
plt.ylabel('Frequency')
plt.show()

# Perform sensitivity analysis
df_rho, df_p = model.spearman_r()

# Print Spearman correlation coefficients for Membrane Area
print("Spearman Correlation for Membrane Area:")
print(df_rho[('-', 'Membrane Area')])

# Plot Tornado plot for sensitivity
bst.plots.plot_spearman_1d(
    df_rho[('-', 'Membrane Area')],
    index=[param.describe() for param in model.parameters],
    name='Membrane Area Sensitivity'
)
#%%
# Define metrics calculation function
def calculate_metrics(j, R, target_ratio):
    # 업데이트된 변수 설정
    energy_cost_per_kwh = 0.1  # USD/kWh
    membrane_cost_per_m2 = 100  # USD/m²
    maintenance_factor = 0.05  # 5% of membrane cost annually
    operational_factor = 0.02  # 2% of total costs
    
    # ED1 설정
    ED1.j = j
    ED1.R = R
    ED1.target_ratio = target_ratio
    ED1.simulate()
    results = ED1.design_results
    
    # 에너지 소비량 계산 (kWh)
    power_consumption_kwh = results['Power consumption'] / 1000  # W -> kW
    power_cost = power_consumption_kwh * energy_cost_per_kwh  # USD
    
    # 멤브레인 비용 계산
    membrane_area = results['Membrane area']  # m²
    membrane_cost = membrane_area * membrane_cost_per_m2  # USD
    
    # 유지 보수 비용 (매년)
    maintenance_cost = membrane_cost * maintenance_factor
    
    # 운영 비용 (운영비의 2%)
    operational_cost = (power_cost + membrane_cost + maintenance_cost) * operational_factor
    
    # 총 비용 계산
    total_cost = power_cost + membrane_cost + maintenance_cost + operational_cost
    
    # VFA 회수량 계산 (kg)
    vfa_recovered_mass = ED1.outs[1].imass['AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid'].sum()
    
    # Cost 및 Specific energy 계산
    cost_per_kg_vfa = total_cost / vfa_recovered_mass
    specific_energy = power_consumption_kwh / vfa_recovered_mass
    
    return cost_per_kg_vfa, specific_energy
#%% Run extended simulation to 168 hours (7 days) to check steady state
extended_time = 168 * 3600  # Total simulation time of 168 hours in seconds
ED1.t = extended_time

# Redefine hypothetical tank volumes for extended simulation
V_dc = inf_dc.F_vol * 24  # Updated total volume in the dilute compartment for 168 hours
V_ac = inf_ac.F_vol * 24  # Updated total volume in the concentrate compartment for 168 hours

# Initialize lists for total VFA concentrations (excluding LacticAcid) over extended time
total_vfa_concentration_dc_ext = []
total_vfa_concentration_ac_ext = []
time_points_ext = range(int(ED1.t))

# Run simulation over extended time to observe steady state behavior
for second in time_points_ext:
    total_vfa_dc = 0
    total_vfa_ac = 0
    
    for ion in ED1.CE_dict:
        if ion != 'LacticAcid':
            n_transferred = ED1.calculate_flux(ED1.j * ED1.A_m).get(ion, 0) * ED1.A_m
            available_amount = inf_dc.imol[ion] * 1e3
            actual_transfer = min(n_transferred, available_amount)
            
            inf_dc.imol[ion] -= actual_transfer / 1e3
            inf_ac.imol[ion] += actual_transfer / 1e3
            
            total_vfa_dc += inf_dc.imol[ion] * 1e3
            total_vfa_ac += inf_ac.imol[ion] * 1e3
    
    # Calculate VFA concentrations in each compartment
    conc_vfa_dc = total_vfa_dc / V_dc
    total_vfa_concentration_dc_ext.append(conc_vfa_dc)
    
    conc_vfa_ac = total_vfa_ac / V_ac
    total_vfa_concentration_ac_ext.append(conc_vfa_ac)

# Convert time_points_ext to hours for plotting
time_points_hours_ext = [t / 3600 for t in time_points_ext]

# Plot the total VFA concentration changes over time in mM for extended simulation
plt.figure(figsize=(7, 5))
plt.plot(time_points_hours_ext, [conc * 1000 for conc in total_vfa_concentration_dc_ext], label="Total VFA (dc)", linestyle='--')
plt.plot(time_points_hours_ext, [conc * 1000 for conc in total_vfa_concentration_ac_ext], label="Total VFA (ac)", linestyle='-')

# Customize plot appearance
plt.xlabel('Time (hr)', fontsize=16, fontweight='bold')  # Set font size and weight
plt.ylabel('Total VFA Concentration (mM)', fontsize=16, fontweight='bold')
# plt.title('Total VFA Concentration Changes in DC and AC Over Extended Time (168 hrs)', fontsize=16, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)  # Set tick label size
plt.show()
#%%
# Define hypothetical tank volumes for concentration calculations
V_dc = inf_dc.F_vol*24  # Total volume in the dilute compartment over the time period
V_ac = inf_ac.F_vol*24  # Total volume in the concentrate compartment over the time period

# Initialize lists to store total VFA concentrations (excluding LacticAcid) over time
total_vfa_concentration_dc = []
total_vfa_concentration_ac = []
time_points = range(int(ED1.t))

# 매 시간마다 각 compartment에서 총 VFA 농도 업데이트
for second in time_points:
    # 매 시간 동안 각 compartment에서 VFA 이동량 계산
    total_vfa_dc = 0
    total_vfa_ac = 0
    
    for ion in ED1.CE_dict:
        if ion != 'LacticAcid':  # LacticAcid는 제외
            # 이온 이동량 계산 (매 시간당 이동량을 기반으로)
            n_transferred = ED1.calculate_flux(ED1.j * ED1.A_m).get(ion, 0) * ED1.A_m  # 각 시간의 이동량
            available_amount = inf_dc.imol[ion] * 1e3
            actual_transfer = min(n_transferred, available_amount)  # 실제 이동량 조절
            
            # eff_dc와 eff_ac의 농도 업데이트
            inf_dc.imol[ion] -= actual_transfer / 1e3  # 희석 compartment에서 감소
            inf_ac.imol[ion] += actual_transfer / 1e3 # 농축 compartment에서 증가
            
            # 희석 compartment의 총 VFA 몰수 합산
            total_vfa_dc += inf_dc.imol[ion] * 1e3
            
            # 농축 compartment의 총 VFA 몰수 합산
            total_vfa_ac += inf_ac.imol[ion] * 1e3
    
    # 희석 compartment의 총 VFA 농도 계산
    conc_vfa_dc = total_vfa_dc / V_dc
    total_vfa_concentration_dc.append(conc_vfa_dc)
    
    # 농축 compartment의 총 VFA 농도 계산
    conc_vfa_ac = total_vfa_ac / V_ac
    total_vfa_concentration_ac.append(conc_vfa_ac)
    
# Convert time_points to hours for plotting
time_points_hours = [t / 3600 for t in time_points]  # Convert seconds to hours

# Plot the total VFA concentration changes over time in mM
plt.figure(figsize=(7, 5))
plt.plot(time_points_hours, [conc * 1000 for conc in total_vfa_concentration_dc], label="Total VFA (dc)", linestyle='--')
plt.plot(time_points_hours, [conc * 1000 for conc in total_vfa_concentration_ac], label="Total VFA (ac)", linestyle='-')

# Customize plot appearance
plt.xlabel('Time (hr)', fontsize=16, fontweight='bold')
plt.ylabel('Total VFA Concentration (mM)', fontsize=16, fontweight='bold')
# plt.title('Total VFA Concentration Changes in DC and AC Over 24 Hours', fontsize=16, fontweight='bold')
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