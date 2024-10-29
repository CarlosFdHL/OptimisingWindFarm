import numpy as np
import pandas as pd

#########################################
# READ DATA FROM FILE

def read_number_in_file(path, start_line):
    """
    File reading function
    Example of file structure: 'n' '=' '2'
        start_line = n
        return 2
    Parameters:
    - path (str): Path to the file
    - start_line (str): Word/s before the number that is wanted
    Returns:
    - number (float): Number that is after start_line and an equal sign
    """
    try:
        with open(path,'r') as file:
            lines = file.readlines()
            original_lines = lines
        found = False
        for line in lines:
            if start_line in line:
                words = line.split()
                if words[0] == str(start_line) and words[1] == "=":
                    found = True
                    return float(words[2])
        if not found:
            print(f"No line starting with '{start_line}' was found")
        file.close()
    except Exception as e:
        print("An error occurred:", e)
        # Restore the original file

#########################################
# FILE LOADING FUNCTIONS

def load_general_data(data_file):
    """
    Loads general data including inflation rate and the number of years for calculations.
    
    Parameters:
    - data_file (str): Path to the data file.

    Returns:
    - inflation_rate (float): The constant inflation rate.
    - n_years (int): Lifetime of the project.
    """
    inflation_rate = read_number_in_file(data_file, 'inflation_rate')  # Assuming constant inflation
    n_years = int(read_number_in_file(data_file, 'n_years'))
    return inflation_rate, n_years

def load_project_data(data_file):
    """
    Loads project-specific data such as CAPEX, OPEX, and installed capacity.

    Parameters:
    - data_file (str): Path to the data file.plot_results()

    Returns:
    - capex_per_MW (float): CAPEX per megawatt of installed capacity.
    - opex_per_MW (float): OPEX per megawatt of installed capacity.
    - installed_capacity_MW (float): Total installed capacity in MW.
    """
    capex_per_MW = read_number_in_file(data_file, 'capex_per_MW')
    opex_per_MW = read_number_in_file(data_file, 'opex')
    installed_capacity_MW = read_number_in_file(data_file, 'installed_capacity')
    return capex_per_MW, opex_per_MW, installed_capacity_MW

def load_loan_data(data_file, loan_options, option):
    """
    Loads loan-specific data such as interest rate and loan term.

    Parameters:
    - data_file (str): Path to the data file.
    - loan_options (pd.DataFrame): DataFrame containing different loan options.

    Returns:
    - interest_rate (float): The annual interest rate for the loan.
    - loan_per_MW (float): The loan amount per MW.
    - loan_years (int): Duration of the loan repayment in years.
    """
    interest_rate = read_number_in_file(data_file, 'interest_rate')
    loan_per_MW = loan_options[option].iloc[0]  # Choosing 'option' option for loan per MW
    loan_years = int(read_number_in_file(data_file, 'loan_repayment_duration'))
    return interest_rate, loan_per_MW, loan_years

def load_tax_data(data_file):
    """
    Loads the tax percentage for the project.

    Parameters:
    - data_file (str): Path to the data file.

    Returns:
    - tax_percentage (float): The percentage of taxes applied to the revenue. [p/u]
    """
    return read_number_in_file(data_file, 'TAX')

def load_energy_data(electricity_file, power_file, N_turbines):
    """
    Loads energy data such as hourly electricity prices and power output.

    Returns:
    - hourly_eprice (numpy array): Array of hourly electricity price forecasts.
    - hourly_poutput (numpy array): Array of hourly power output values.
    """
    hourly_eprice = pd.read_csv(electricity_file, header=None).to_numpy().flatten()
    hourly_poutput = pd.read_csv(power_file, header=None).to_numpy().flatten().T * N_turbines
    return hourly_eprice, hourly_poutput
