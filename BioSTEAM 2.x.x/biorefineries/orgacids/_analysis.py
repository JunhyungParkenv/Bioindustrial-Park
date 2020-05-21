#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:24:42 2020

@author: yalinli_cabbi
"""

'''
TODO:    
    Consider evaluating across the coordinate of:
        Heavy organic acid content in feedstock/fermentation broth
        Purity of lactic acid product
    (should use evaluate_across_coordinate if the variable affects system simulation)
    
    If having multiple coordinates, consider summarizing all percentiles into
        one df and make percentiles as the columns, make multi-level column labels
'''


# %% Setup

import numpy as np
import pandas as pd

percentiles = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]


# %% Evaluate and calculate probabilities

from orgacids.model import orgacids_model

'''Quick look at baseline values'''
# Set seed to make sure each time the same set of random numbers will be used
np.random.seed(3221)
N_simulation = 10
samples = orgacids_model.sample(N=N_simulation, rule='L')
orgacids_model.load_samples(samples)
baseline = orgacids_model.metrics_at_baseline()
baseline_df = pd.DataFrame(data=np.array([[i for i in baseline.values()],]), 
                            index=('baseline',), columns=baseline.keys())
baseline_df.to_excel('baseline.xlsx')

'''Full evaluation'''
orgacids_model.evaluate()
# Parameters and probabilities
parameter_len = len(orgacids_model.get_baseline_sample())
parameters = orgacids_model.table.iloc[:, :parameter_len].copy()
# Add baseline values to the end
parameters.loc['baseline'] = orgacids_model.get_baseline_sample()

Monte_Carlo_results = orgacids_model.table.iloc[:, parameter_len::].copy()
Monte_Carlo_percentiles = Monte_Carlo_results.quantile(q=percentiles)
# Add baseline values to the end
Monte_Carlo_results.loc['baseline'] = orgacids_model.metrics_at_baseline()

# Note that if only one metric is used, then need to make sure it's a tuple
spearman_metrics = orgacids_model.metrics[0:4]
spearman_results = orgacids_model.spearman(spearman_metrics)

# Calculate the probabilities of each parameter and the overall scenario
probabilities = {}
for i in range(parameter_len):
    p = orgacids_model.get_parameters()[i]
    p_values = parameters.iloc[:, 2*i]
    #!!! cdf vs. 1-cdf, might need to be mannually reviewed
    if spearman_results.iloc[:, 0][i]>0:
        probabilities[p.name] = p.distribution.cdf(p_values)
    else:
        probabilities[p.name] = np.ones(len(p_values))-p.distribution.cdf(p_values)
    parameters.insert(loc=2*i+1, 
                      column=(parameters.iloc[:, 2*i].name[0], 'Probability'), 
                      value=probabilities[p.name],
                      allow_duplicates=True)

'''Output to Excel'''
with pd.ExcelWriter('Evaluation scenarios and probabilities.xlsx') as writer:
    parameters.to_excel(writer, sheet_name='Parameters')
    Monte_Carlo_results.to_excel(writer, sheet_name='Monte Carlo')
    Monte_Carlo_percentiles.to_excel(writer, sheet_name='Monte Carlo Percentiles')
    spearman_results.to_excel(writer, sheet_name='Spearman')


# %% Monte Carlo and evaluate across lactic acid yield

from chaospy import distributions as shape
from orgacids.model import orgacids_model_LA_yield, set_LA_yield

'''Evaluate'''
np.random.seed(3221)
N_simulation = 10
LA_yield_samples = orgacids_model_LA_yield.sample(N=N_simulation, rule='L')
orgacids_model_LA_yield.load_samples(LA_yield_samples)

#!!! Should be 0.93 to 0.55, now temporarily only to 0.65 due to the F_pre_S404 bug
LA_yield_coordinate = np.linspace(0.93, 0.65, 28*1+1)
LA_yield_distribution = shape.Triangle(0.55, 0.76, 0.93)
LA_yield_probability = np.ones(len(LA_yield_coordinate)) - \
    LA_yield_distribution.cdf(LA_yield_coordinate)

LA_yield_data = orgacids_model_LA_yield.evaluate_across_coordinate(
    'Lactic acid yield', set_LA_yield, LA_yield_coordinate, notify=True)

columns = pd.MultiIndex.from_arrays([LA_yield_coordinate, LA_yield_probability],
                                    names=['Lactic acid yield', 'Probability'])

LA_yield_MSP = pd.DataFrame(
    data=LA_yield_data[('Biorefinery', 'Minimum selling price [$/kg]')],
    columns=columns)

LA_yield_MSP_percentiles = LA_yield_MSP.quantile(q=percentiles)

# Frequency of simulation that has MSP < $3.5/kg
LA_yield_MSP_f = LA_yield_MSP[LA_yield_MSP<3.5].count()/N_simulation
LA_yield_MSP_f.name = 'frequency'
LA_yield_MSP = LA_yield_MSP.append(LA_yield_MSP_f)

'''Output to Excel'''
with pd.ExcelWriter('Evaluation across lactic acid yield.xlsx') as writer:
    LA_yield_MSP.to_excel(writer, sheet_name='LA yield')
    LA_yield_MSP_percentiles.to_excel(writer, sheet_name='LA yield percentiles')


# %% Feedstock price and carbohydrate content

from orgacids.model import orgacids_model_feedstock, set_feedstock_carbs, prices

'''Evaluate'''
np.random.seed(3221)
N_simulation = 1
feedstock_samples_1d = orgacids_model_feedstock.sample(N=N_simulation, rule='L')
feedstock_samples = feedstock_samples_1d[:, np.newaxis]
orgacids_model_feedstock.load_samples(feedstock_samples)

feedstock_carbs_coordinate = np.linspace(0.7, 0.4, 30*1+1)

feedstock_data = orgacids_model_feedstock.evaluate_across_coordinate(
    'Feedstock carbohydate content', set_feedstock_carbs, 
    feedstock_carbs_coordinate, notify=True)

feedstock_MSP = pd.DataFrame({
    ('Parameter','Carbohydrate content [dry mass %]'): feedstock_carbs_coordinate
    })

for (i, j) in zip(feedstock_data.keys(), feedstock_data.values()):
    feedstock_MSP[i] = j[0]

'''Organize data for easy plotting'''
x_axis = [f'{i:.2f}' for i in feedstock_carbs_coordinate]
x_axis *= len(prices)
y_axis = sum(([f'{i:.0f}']*len(feedstock_carbs_coordinate) for i in prices), [])

MSP = []
NPV = []
for i in range(feedstock_MSP.columns.shape[0]):
    if 'MSP' in feedstock_MSP.columns[i][1]:
        MSP +=  feedstock_MSP[feedstock_MSP.columns[i]].to_list()
    if 'NPV' in feedstock_MSP.columns[i][1]:
        NPV +=  feedstock_MSP[feedstock_MSP.columns[i]].to_list()

feedstock_MSP_plot = pd.DataFrame()
feedstock_MSP_plot['Carbohydrate content [dry mass %]'] = x_axis
feedstock_MSP_plot['Price [$/dry-ton]'] = y_axis
feedstock_MSP_plot['MSP [$/kg]'] = MSP
feedstock_MSP_plot['NPV [$]'] = NPV

'''Output to Excel'''
name = 'Evaluation across feedstock cost and carbohydrate content.xlsx'
with pd.ExcelWriter(name) as writer:
    feedstock_MSP.to_excel(writer, sheet_name='Evaluation data')
    feedstock_MSP_plot.to_excel(writer, sheet_name='For plotting')


# %% Evaluating across internal rate of return

from orgacids.model import orgacids_model_IRR

'''
Note:
    Not using `evaluate_across_coordinate` function as changing IRR
    does not affect the system, using IRR as the metrics for evaluation 
    will save considerable time.
'''

'''Evaluate'''
np.random.seed(3221)
N_simulation = 10
samples = orgacids_model_IRR.sample(N=N_simulation, rule='L')
orgacids_model_IRR.load_samples(samples)
orgacids_model_IRR.evaluate()

parameter_len = len(orgacids_model_IRR.get_baseline_sample())
IRR_results = orgacids_model_IRR.table.iloc[:, parameter_len::].copy()
IRR_percentiles = IRR_results.quantile(q=percentiles)

'''To get a quick plot'''
from biosteam.evaluation.evaluation_tools import plot_montecarlo_across_coordinate
from orgacids.model import IRRs
MSP_indices = [metric.index for metric in orgacids_model_IRR.metrics
                if 'NPV' not in metric.index[1]]
MSP_data = orgacids_model_IRR.table[MSP_indices]
plot_montecarlo_across_coordinate(IRRs, MSP_data.values)

'''Output to Excel'''
with pd.ExcelWriter('Evaluation across IRR.xlsx') as writer:
    IRR_results.to_excel(writer, sheet_name='IRR')
    IRR_percentiles.to_excel(writer, sheet_name='IRR Percentiles')


# %% Backup codes for calculating and inserting scenario probability

# # Calculate the overall probabilities of scenario
# probability_list = list(i for i in probabilities.values())
# scenario_probability = probability_list[0].copy()
# for i in range(parameter_len-1):
#     scenario_probability *= list(i for i in probabilities.values())[i+1]
# probabilities['scenario_abs'] = scenario_probability

# # Normalize scenario probabilities by baseline probability
# scenario_probability_normalzied = probabilities['scenario_abs'].copy()
# scenario_probability_normalzied /= \
#     np.ones(len(probabilities['scenario_abs']))*probabilities['scenario_abs'][-1]
# probabilities['scenario_normalized'] = scenario_probability_normalzied
# parameters.insert(loc=parameters.shape[1],
#                   column=('Scenario', 'Absolute probability'), 
#                   value=probabilities['scenario_abs'])
# parameters.insert(loc=parameters.shape[1],
#                   column=('Scenario', 'Normalized probability'), 
#                   value=probabilities['scenario_normalized'])

# # Add scenario probabilities to Monte Carlo results
# Monte_Carlo_results.insert(loc=0, column=('Scenario', 'Normalized probability'),
#                             value=parameters.iloc[:, parameters.shape[1]-1])








