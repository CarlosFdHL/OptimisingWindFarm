import numpy as np
import pandas as pd
import numpy_financial as npf
import sys
from load_data_functions import *

#########################################
# CALCULATION FUNCTIONS

def calculate_npv(installed_capacity_MW, data_file, loan_options, electricity_price_file, power_output_file):
    """
    Calculates the Net Present Value (NPV) of the project.

    Parameters:
    - installed_capacity_MW (float): Installed capacity in MW.
    - data_file (str): Path to the file containing financial data.
    - loan_options (pd.DataFrame): DataFrame containing different loan options.
    - electricity_price_file (str): Path to the electricity price forecast CSV file.
    - power_output_file (str): Path to the power output CSV file.

    Returns:
    - npv (float): Net Present Value of the project.
    """
    
    # Load general data
    inflation_rate, n_years = load_general_data(data_file)
    
    # Load project data
    capex_per_MW, opex_per_MW, _ = load_project_data(data_file) #Also have the option to read from file the installed capacity
    
    # Load loan data
    interest_rate, loan_per_MW, loan_years = load_loan_data(data_file, loan_options, option = 'LOW')
    
    # Load tax data
    tax_percentage = load_tax_data(data_file)
    
    # Load energy data
    hourly_eprice, hourly_poutput = load_energy_data(electricity_price_file, power_output_file)

    # Calculate inflation matrix
    inflation_matrix = calculate_inflation_matrix(inflation_rate, n_years)

    # Calculate loan vector
    loan = loan_per_MW * installed_capacity_MW
    loan_vector = calculate_loan(loan, interest_rate, loan_years, n_years, inflation_matrix)

    # Calculate nominal revenue
    nominal_revenue = calculate_nominal_revenue(hourly_eprice, hourly_poutput, n_years, inflation_matrix)

    # Calculate costs vector
    costs_vector = calculate_costs_vector(loan_vector, opex_per_MW, installed_capacity_MW, n_years, inflation_matrix)

    # Calculate amortization vector
    amortization_vector = calculate_amortization_vector(capex_per_MW, installed_capacity_MW, n_years)

    # Calculate EBITDA
    ebitda = nominal_revenue - costs_vector

    # Calculate taxes
    taxes_vector = -(ebitda - amortization_vector) * tax_percentage

    # Calculate free cash flow
    free_cash_flow = ebitda + taxes_vector + amortization_vector
    free_cash_flow = np.insert(free_cash_flow, 0, -capex_per_MW * installed_capacity_MW)

    # Calculate NPV
    npv = npf.npv(interest_rate, free_cash_flow)
    
    return npv

def calculate_inflation_matrix(inflation_rate, n_years):
    """
    Generates a diagonal inflation matrix for adjusting financial figures over time.

    Parameters:
    - inflation_rate (float): The constant inflation rate.
    - n_years (int): The duration of the project.

    Returns:
    - inflation_matrix (numpy array): Diagonal matrix accounting for inflation over time.
    """
    A = np.array([(1 + inflation_rate) ** k for k in range(n_years)])
    return np.diag(A)

def calculate_loan(loan_amount, interest_rate, loan_term, total_years, inflation_adjustment_matrix):
    """
    Function to calculate the inflation-adjusted loan repayment vector.

    Parameters:
    - loan_amount (float): Total amount of the loan.
    - interest_rate (float): Annual interest rate of the loan.
    - loan_term (int): Duration of the loan in years.
    - total_years (int): Total number of years for the repayment of the loan.
    - inflation_adjustment_matrix (numpy array): Matrix to adjust repayments for inflation.
    
    Returns:
    - adjusted_loan_repayment_vector (numpy array): Loan repayment vector adjusted for inflation.
    """
    # Calculate the annual loan repayment amount
    annual_repayment = loan_amount * interest_rate / (1 - (1 + interest_rate) ** (-loan_term))

    # Initialize the repayment schedule
    annual_repayment_schedule = (annual_repayment * np.ones(shape=(loan_term, 1))).flatten()
    loan_repayment_vector = np.zeros(shape=(total_years,))
    loan_repayment_vector[:loan_term] = -annual_repayment_schedule
    adjusted_loan_repayment_vector = np.dot(inflation_adjustment_matrix, loan_repayment_vector)
    
    return adjusted_loan_repayment_vector

def calculate_nominal_revenue(hourly_eprice, hourly_poutput, n_years, inflation_matrix):
    """
    Calculates the nominal revenue adjusted for inflation.

    Parameters:
    - hourly_eprice (numpy array): Array of hourly electricity prices.
    - hourly_poutput (numpy array): Array of hourly power outputs.
    - n_years (int): The number of years for the revenue calculation.
    - inflation_matrix (numpy array): The inflation matrix to adjust revenue.

    Returns:
    - nominal_revenue (numpy array): Array of nominal revenues for each year.
    """
    real_erevenue = np.dot(hourly_eprice, hourly_poutput) / 10**6  # Revenue in MDKK
    nominal_revenue = np.ones(shape=(n_years,)) * real_erevenue
    return np.dot(nominal_revenue, inflation_matrix)

def calculate_costs_vector(loan_vector, opex_per_MW, installed_capacity_MW, n_years, inflation_matrix):
    """
    Calculates the total cost vector including loan repayments and OPEX.

    Parameters:
    - loan_vector (numpy array): Array representing loan repayment values.
    - opex_per_MW (float): Operational expenses per MW of capacity.
    - installed_capacity_MW (float): Installed capacity in MW.
    - n_years (int): The number of years for the cost calculation.
    - inflation_matrix (numpy array): The inflation matrix to adjust costs.

    Returns:
    - costs_vector (numpy array): Array representing the total costs for each year.
    """
    opex_MDKK = -opex_per_MW * installed_capacity_MW  # OPEX in MDKK per year
    opex_vector = np.dot(inflation_matrix, np.ones((n_years,)) * opex_MDKK)
    return -(loan_vector + opex_vector)

def calculate_amortization_vector(capex_per_MW, installed_capacity_MW, n_years):
    """
    Calculates the amortization vector over the given years.

    Parameters:
    - capex_per_MW (float): Capital expenditures per MW.
    - installed_capacity_MW (float): Total installed capacity in MW.
    - n_years (int): The number of years for amortization.

    Returns:
    - amortization_vector (numpy array): Array representing amortization for each year.
    """
    amortization = capex_per_MW * installed_capacity_MW / n_years
    return np.ones((n_years,)) * amortization


if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except ValueError:
            print("Error: The value of N must be an integer.")
            sys.exit(1)
    else:
        print("Error: You must provide a number of turbines for the calculation.")
        sys.exit(1)

    installed_capacity_MW = N * 14
    loan_options = pd.DataFrame(data = {'LOW': [14.82967991], 'MED': [23.12919455], 'HIGH': [31.42870918]})
    output = calculate_npv(installed_capacity_MW, data_file="data.txt", loan_options=loan_options, 
                         electricity_price_file="electricity_price_forecast.csv", power_output_file="power_output.csv")
    print(f'The NPV for {N} turbines is: {output} MDKK')

