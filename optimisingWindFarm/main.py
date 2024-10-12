import numpy_financial as npf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scripts.calculations_functions import *
from scripts.plot_functions import *
#from scripts.load_data_functions import *

loan_options = pd.DataFrame(data = {'LOW': [14.82967991], 'MED': [23.12919455], 'HIGH': [31.42870918]})

def store(N):

    installed_capacity = N[0] * 14  
    npv = -calculate_N_turbines(N[0]) 
    
    hist_MW.append(installed_capacity)
    hist_npv.append(npv)
    return hist_MW, hist_npv


def calculate_N_turbines(N):
    if isinstance(N, np.ndarray):
        N = N[0]

    installed_capacity_MW = int(np.round(N)) * 14
    output = calculate_npv(installed_capacity_MW, data_file="data/data.txt", loan_options=loan_options, 
                         electricity_price_file="data/electricity_price_forecast.csv", power_output_file="data/power_output.csv")
    print('N: ', N, 'NPV: ', output)
    return -output

# Set the bounds:
# 1. Installed capacity should not be lower than 800 or greater than 1200
#    This translates to 800 <= N * 14 <= 1200, i.e., 800/14 <= N <= 1200/14
# 2. Installed capacity divided by 7.5 should not be greater than 160
#    This translates to N * 14 / 7.5 <= 160, i.e., N <= 160 * 7.5 / 14
bounds = [(800 / 14, min(1200 / 14, 160 * 7.5 / 14))]

x0 = 58  # Initial value for N

result = minimize(calculate_N_turbines, x0=x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 20}, callback = store)

# Print the results
print("Optimal value of N:", result.x)
print("Optimal installed capacity (MW):", -result.x * 14)
print("Optimal value of the NPV:", result.fun)
print("Number of iterations:", result.nit)

plot_results()
