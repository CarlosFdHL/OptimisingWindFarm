import numpy as np
import pandas as pd
import numpy_financial as npf
import sys
from scripts.load_data_functions import *

#########################################
# CALCULATION FUNCTIONS

opex_options = pd.DataFrame(data = {'LOW': [0.635], 'MED': [0.7], 'HIGH': [0.764]}) #OPEX options in MDKK/MW

def calculate_npv(installed_capacity_MW, data_file, loan_options, electricity_price_file, 
                  power_output_file, loan_option, extra_income, print_option = 0):
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
    
    N = installed_capacity_MW / 14

    # Load general data
    inflation_rate, n_years = load_general_data(data_file)
    
    # Load project data
    _, _, _, wacc = load_project_data(data_file) # Also have the option to read from file the installed capacity
    opex_per_MW = opex_options[loan_option].iloc[0]
    # Load loan data
    interest_rate, loan_per_MW, loan_years, loan_percentage = load_loan_data(data_file, loan_options, option = loan_option)
    
    # Load tax data
    tax_rate = load_tax_data(data_file)
    
    # Load energy data
    hourly_eprice, hourly_poutput = load_energy_data(electricity_price_file, power_output_file, N)

    # Calculate inflation matrix
    inflation_matrix = calculate_inflation_matrix(inflation_rate, n_years)

    # Calculate loan vector
    loan = loan_per_MW * installed_capacity_MW
    capex_per_MW = loan_per_MW
    loan = loan * loan_percentage
    loan_vector, interest_payment, debt_payment = calculate_loan(loan, interest_rate, loan_years, n_years, inflation_matrix)

    # Calculate nominal revenue
    nominal_revenue = calculate_nominal_revenue(hourly_eprice, hourly_poutput, n_years, inflation_matrix, extra_income)

    # Check if the investment passes the DSCR test
    if print_option:
        DSCR_condition = True
        idx_array = []
        for i, (revenue, loan) in enumerate(zip(nominal_revenue, loan_vector)):
            if abs(revenue/loan) < 1.2:
                DSCR_condition = False
                idx_array.append(i)

        if DSCR_condition == True:
            print("\nThis investment passess de DSCR test")
        else:
            print(f"\nThis investment does NOT pass the DSCR test in years : {idx_array}")
            print(nominal_revenue)
            print(loan_vector)

    # Calculate costs vector
    costs_vector = calculate_costs_vector(loan_vector, opex_per_MW, installed_capacity_MW, n_years, inflation_matrix)

    # Calculate amortization vector
    amortization_vector = calculate_amortization_vector(capex_per_MW, installed_capacity_MW, n_years)

    # Calculate EBITDA
    ebitda = nominal_revenue + costs_vector

    # Calculate taxes
    taxes_vector = calculate_taxes(n_years, ebitda, amortization_vector, interest_payment, tax_rate)

    # Calculate free cash flow
    free_cash_flow = ebitda + taxes_vector
    free_cash_flow = np.insert(free_cash_flow, 0, -capex_per_MW * installed_capacity_MW)

    # Calculate NPV
    npv = npf.npv(wacc, free_cash_flow)

    # Calculate IRR
    irr = npf.irr(free_cash_flow)

    if print_option:
        print(free_cash_flow)
        print(f"IRR: {round(irr, 5)}")
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
    - total_years (int): Total lifespan of the project.
    - inflation_adjustment_matrix (numpy array): Matrix to adjust repayments for inflation.
    
    Returns:
    - adjusted_loan_repayment_vector (numpy array): Loan repayment vector with size n_years.
    """
    # Calculate the annual loan repayment amount
    annual_repayment = npf.pmt(interest_rate, loan_term, loan_amount)

    # Initialize the repayment schedule
    annual_repayment_schedule = (annual_repayment * np.ones(shape=(loan_term, 1))).flatten()
    loan_repayment_vector = np.zeros(shape=(total_years,))
    loan_repayment_vector[:loan_term] = -annual_repayment_schedule

    # Calculate interest and debt payments
    interest_payment = np.zeros(shape=(total_years,))
    debt_payment = np.zeros(shape=(total_years,))

    for i in range(total_years):
        interest_payment[i] = loan_amount * interest_rate
        debt_payment[i] = -(abs(annual_repayment) - abs(interest_payment[i]))
        loan_amount -= debt_payment[i]


    return loan_repayment_vector, interest_payment, debt_payment

def calculate_nominal_revenue(hourly_eprice, hourly_poutput, n_years, inflation_matrix, extra_income):
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
    nominal_revenue = np.dot(nominal_revenue, inflation_matrix) + np.full_like(nominal_revenue, extra_income) #Add extra income as a fixed value over the years

    return nominal_revenue

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
    return (-loan_vector + opex_vector)

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
    

def calculate_taxes(n_years, ebitda, amortization_vector, interest_payment, tax_rate):
    # Earnings before interest and taxes (EBIT)
    ebit = ebitda - amortization_vector
    # Earnings before taxes (EBT)
    ebt = ebit - interest_payment

    # Initialize loss carryforward vectors
    loss_carryforward_boy = np.zeros(n_years)
    loss_carryforward_eoy = np.zeros(n_years)

    # Calculate loss carryforward for each year
    for i in range(n_years - 1):
        if ebt[i] < 0:
            loss_carryforward_eoy[i] = ebt[i]  # Apply full EBT as carryforward loss if it's negative
        else:
            loss_carryforward_eoy[i] = 0
        loss_carryforward_boy[i + 1] = loss_carryforward_eoy[i]  # Transfer end-of-year loss to next year's beginning-of-year

    # Adjust EBT after applying loss carryforward
    ebt_after_loss_carryforward = ebt + loss_carryforward_eoy

    # Calculate taxes
    taxes_vector = np.where(ebt_after_loss_carryforward > 0, -ebt_after_loss_carryforward * tax_rate, 0)

    return taxes_vector

def calculate_extra_income(extra_income, installed_capacity_MW, data_file, loan_options, electricity_price_file, power_output_file, loan_option):
    return calculate_npv(installed_capacity_MW, data_file, loan_options, electricity_price_file, power_output_file, loan_option, extra_income)
