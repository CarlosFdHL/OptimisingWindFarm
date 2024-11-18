import numpy_financial as npf
import numpy as np
import pandas as pd
from scipy.optimize import minimize, fsolve

from scripts.calculations_functions import *
from scripts.plot_functions import *
#from scripts.load_data_functions import *

import warnings
warnings.filterwarnings("ignore") #Ignore warnings

loan_options = pd.DataFrame(data = {'LOW': [12.4905691], 'MED': [18.9896121], 'HIGH': [25.4886551]}) #Loan options in MDKK/MW
decomitioning_options = pd.DataFrame(data = {'LOW': [1.744], 'MED': [3.488], 'HIGH': [5.233]})
opex_options = pd.DataFrame(data = {'LOW': [0.635], 'MED': [0.7], 'HIGH': [764]})
loan_option = 'MED'

def store(N):

    installed_capacity = N[0] * 14  
    npv = -calculate_N_turbines(N[0]) 
    
    hist_MW.append(installed_capacity)
    hist_npv.append(npv)
    return hist_MW, hist_npv


def calculate_N_turbines(N):
    if isinstance(N, np.ndarray):
        N = N[0]
    N = int(N)
    installed_capacity_MW = N * 14
    output = calculate_npv(installed_capacity_MW, data_file="data/data.txt", loan_options=loan_options, 
                         electricity_price_file="data/electricity_price_forecast.csv", power_output_file="data/power_output.csv",
                         loan_option=loan_option, opex_options = opex_options, decomitioning_options = decomitioning_options, extra_income = 0)
    print('N: ', N, 'Installed capacity: ', installed_capacity_MW, 'NPV: ', output)

    return -output


# Set the bounds:
# 1. Installed capacity should not be lower than 800 or greater than 1200
#    This translates to 800 <= N * 14 <= 1200, i.e., 800/14 <= N <= 1200/14
# 2. Installed capacity divided by 7.5 should not be greater than 160
#    This translates to N * 14 / 7.5 <= 160, i.e., N <= 160 * 7.5 / 14
bounds = [(800 / 14, 994/14)]

x0 = 58  # Initial value for N

result = minimize(calculate_N_turbines, x0=x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 20}, callback = store) #Obtain max profit


NPV_objective = read_number_in_file(path  = "data/data.txt", start_line = "NPV_objective")




# Print the results
calculate_npv(result.x * 14, data_file="data/data.txt", loan_options=loan_options, 
                         electricity_price_file="data/electricity_price_forecast.csv", power_output_file="data/power_output.csv",
                         loan_option=loan_option, opex_options = opex_options, decomitioning_options = decomitioning_options, extra_income = 0, print_option=1)
print("Optimal value of N:", int(result.x))
print("Optimal installed capacity :", int(result.x) * 14, "MW")
print("Optimal value of the NPV:", -result.fun, "MDKK")
print("Number of iterations:", result.nit)

if (-result.fun < NPV_objective):
        extra_income = fsolve(calculate_extra_income, NPV_objective, args=(-result.x * 14, "data/data.txt", loan_options, 
                            "data/electricity_price_forecast.csv", "data/power_output.csv",
                            loan_option))
        extra_income = - extra_income
        print(f"\nThe extra income needed per year to obtain an NPV of {NPV_objective} is {float(extra_income)} MDKK, that results in {float(extra_income/30)} MDKK per year.")

plot_results()


#DSCR = Net income per year / Loan payment

#Needed subsidy from gobernment to get 0 npv
