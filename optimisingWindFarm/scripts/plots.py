import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

def plot_power_output():
    data = pd.read_csv('../data/power_output.csv', header=None)

    power_output = data.iloc[0].values 
    print("data: ", power_output)

    time = pd.date_range(start='2023-01-01', periods=len(power_output), freq='H')

    plt.figure(figsize=(12, 6))
    plt.plot(time, power_output, label='Forecasted Power Output')

    plt.title('Wind Turbine Hourly Power Output Forecast')
    plt.xlabel('Time')
    plt.ylabel('Power Output (MW)')
    plt.grid(True)
    #plt.legend()

    #plt.savefig('../figures/power_output.jpg', format='jpg', dpi=300)

def plot_revenue():
    # Read the power output and electricity price data
    power_data = pd.read_csv('../data/power_output.csv', header=None)
    power_output = power_data.iloc[0].values  

    price_data = pd.read_csv('../data/electricity_price_forecast.csv', header=None)
    electricity_price = price_data[0].values  

    # Ensure both vectors have the same length
    if len(power_output) != len(electricity_price):
        print("Shape of power_output: ", len(power_output))
        print("Shape of electricity_price: ", len(electricity_price))
        raise ValueError("The power output and electricity price data must have the same length.")

    # Calculate revenue in millions of DKK
    revenue = (power_output * electricity_price) / 1e6  

    # Create a time range for the x-axis, assuming hourly data for one year (8760 hours)
    time = pd.date_range(start='2023-01-01', periods=len(power_output), freq='H')

    # Create the figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # First subplot: Forecasted Power Output
    axs[0].plot(time, power_output, color='blue')
    axs[0].set_ylabel('Power Output (kW)', color='blue')
    axs[0].tick_params(axis='y', labelcolor='blue')
    axs[0].set_title('Forecasted Energy Generation (MWh)')
    axs[0].grid(True)

    # Second subplot: Electricity Price Forecast
    axs[1].plot(time, electricity_price, color='green')
    axs[1].set_ylabel('Electricity Price (DKK/MWh)', color='green')
    axs[1].tick_params(axis='y', labelcolor='green')
    axs[1].set_title('Electricity Price Forecast (DKK/MWh)')
    axs[1].set_ylim(-10, 3000)
    axs[1].grid(True)

    # Third subplot: Revenue
    axs[2].plot(time, revenue, color='red')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Revenue [MDKK]', color='red')
    axs[2].tick_params(axis='y', labelcolor='red')
    axs[2].set_title('Revenue [MDKK]')
    axs[2].set_ylim(-0.001, 0.025)
    axs[2].grid(True)

    # Set date format for the x-axis to show only month and day
    date_format = mdates.DateFormatter('%m-%d')
    axs[2].xaxis.set_major_formatter(date_format)

    # Layout adjustment and save the figure
    fig.suptitle('Wind Turbine Power Output, Electricity Price Forecast, and Revenue', y=1.02)
    fig.tight_layout()  
    plt.savefig('../figures/revenue.jpg', format='jpg', dpi=300)


def plot_sensitivity1_NPV():

    file_path = '../data/sensitivity_1_NPV.xlsx'
    excel_data = pd.ExcelFile(file_path)
    df = excel_data.parse('Sheet1')
    df = df.rename(columns={'Unnamed: 0': 'Row'}).set_index('Row')

    x = df.index.astype(float)  
    y = df.columns.astype(float)  
    X, Y = np.meshgrid(x, y)
    Z = df.values.T  

    # Convertir a porcentaje
    X_percentage = X * 100
    Y_percentage = Y * 100
    X_complement = 100 - X_percentage

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X_percentage, Y_percentage, Z, cmap='viridis', edgecolor='k')

    ax.set_xlabel('Share of Debt (%)', fontsize=12, labelpad=15)
    ax.set_ylabel('Interest Rate of Loan (%)', fontsize=12, labelpad=15)
    ax.set_zlabel('NPV [MDKK]', fontsize=12, labelpad=15)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    # Añadir anotaciones en paralelo al eje "Share of Debt (%)" como "100 - Share of Debt (%)"
    for idx, tick in enumerate(ax.get_xticks()):
        print("tick: ", tick)
        if tick>=50 and idx != 7:
            complement_tick = 100 - tick
            ax.text(tick, max(Y_percentage.flatten())-5, Z.max()-1750, f'{complement_tick:.0f}%', 
                color="red", ha="center", va="center", rotation=0, fontsize=12)


    # Añadir etiqueta para indicar el eje complementario
    ax.text(105, max(Y_percentage.flatten())-4.5, Z.min()-200, 'Share of Equity (%)', 
            color="red", ha="center", va="center", rotation=0, fontsize=12)


def plot_sensitivity1_IRR():

    file_path = '../data/sensitivity_1_IRR.xlsx'
    excel_data = pd.ExcelFile(file_path)
    df = excel_data.parse('Sheet1')
    df = df.rename(columns={'Unnamed: 0': 'Row'}).set_index('Row')

    x = df.index.astype(float)  
    y = df.columns.astype(float)  
    X, Y = np.meshgrid(x, y)
    Z = df.values.T  * 100

    # Convertir a porcentaje
    X_percentage = X * 100
    Y_percentage = Y * 100
    X_complement = 100 - X_percentage

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X_percentage, Y_percentage, Z, cmap='plasma', edgecolor='k')

    ax.set_xlabel('Share of Debt (%)', fontsize=12, labelpad=15)
    ax.set_ylabel('Interest Rate of Loan (%)', fontsize=12, labelpad=15)
    ax.set_zlabel('IRR [%]', fontsize=12, labelpad=15)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    # Añadir anotaciones en paralelo al eje "Share of Debt (%)" como "100 - Share of Debt (%)"
    for idx, tick in enumerate(ax.get_xticks()):
        if tick>=50 and idx != 7:
            complement_tick = 100 - tick
            ax.text(tick, max(Y_percentage.flatten())-5, Z.max()- 0.0105*100, f'{complement_tick:.0f}%', 
                color="red", ha="center", va="center", rotation=0, fontsize=12)


    # Añadir etiqueta para indicar el eje complementario
    ax.text(105, max(Y_percentage.flatten())-4.5, Z.min() - 0.0011*100, 'Share of Equity (%)', 
            color="red", ha="center", va="center", rotation=0, fontsize=12)
    








def plot_sensitivity1_CAPEX_OPEX_LOAN(path):
    # Load the Excel files
    capex_df = pd.read_excel(path + 'sensitivity_capex.xlsx', header=None)
    opex_df = pd.read_excel(path + 'sensitivity_opex.xlsx', header=None)
    loan_df = pd.read_excel(path + 'sensitivity_loan.xlsx', header=None)

    # Prepare data for plotting
    scenarios = ['LOW', 'MED', 'HIGH']
    loan_durations = [5, 10, 15, 20, 25, 30]

    # Extract data for plotting
    capex_npv, capex_irr = capex_df.iloc[:, 0], capex_df.iloc[:, 1]
    opex_npv, opex_irr = opex_df.iloc[:, 0], opex_df.iloc[:, 1]
    loan_npv, loan_irr = loan_df.iloc[:, 1], loan_df.iloc[:, 2]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # Set limits manually to align zeros
    npv_min = min(capex_npv.min(), opex_npv.min(), loan_npv.min(), 0) - 500
    npv_max = max(capex_npv.max(), opex_npv.max(), loan_npv.max()) + 1000
    irr_min = npv_min / 800  # Escalar irr para que esté alineado con npv en la posición 0
    irr_max = npv_max / 800   # Ajuste para que IRR y NPV tengan el mismo "peso visual"

    plt.rc('font', family='serif', size=14)  # Set font style to serif and size to 14

    # First subplot for Capex sensitivity with secondary axis
    ax1 = axes[0]
    capex_bars = ax1.bar(scenarios, capex_npv, label='NPV (MDKK)', color='skyblue', width=0.4)
    ax1.set_ylim(npv_min, npv_max)
    ax1.set_ylabel('NPV (MDKK)', fontdict={'size': 14, 'family': 'serif'})
    ax2 = ax1.twinx()
    capex_irr_bars = ax2.bar(scenarios, capex_irr, label='IRR (%)', color='salmon', width=0.2, alpha=0.7)
    ax2.set_ylim(irr_min, irr_max)
    ax2.set_ylabel('IRR (%)', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_title('Sensitivity to Capex', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_xlabel('Scenarios', fontdict={'size': 14, 'family': 'serif'})

    # Second subplot for Opex sensitivity with secondary axis
    ax1 = axes[1]
    opex_bars = ax1.bar(scenarios, opex_npv, label='NPV (MDKK)', color='skyblue', width=0.4)
    ax1.set_ylim(npv_min, npv_max)
    ax1.set_ylabel('NPV (MDKK)', fontdict={'size': 14, 'family': 'serif'})
    ax2 = ax1.twinx()
    opex_irr_bars = ax2.bar(scenarios, opex_irr, label='IRR (%)', color='salmon', width=0.2, alpha=0.7)
    ax2.set_ylim(irr_min, irr_max)
    ax2.set_ylabel('IRR (%)', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_title('Sensitivity to Opex', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_xlabel('Scenarios', fontdict={'size': 14, 'family': 'serif'})

    # Third subplot for Loan duration sensitivity with secondary axis (thicker blue bars)
    ax1 = axes[2]
    loan_bars = ax1.bar(loan_durations, loan_npv, label='NPV (MDKK)', color='skyblue', width=2)  # Increased width for thicker blue bars
    ax1.set_ylim(npv_min, npv_max)
    ax1.set_ylabel('NPV (MDKK)', fontdict={'size': 14, 'family': 'serif'})
    ax2 = ax1.twinx()
    loan_irr_bars = ax2.bar(loan_durations, loan_irr, label='IRR (%)', color='salmon', width=1, alpha=0.7)  # Same width for IRR bars
    ax2.set_ylim(irr_min, irr_max)
    ax2.set_ylabel('IRR (%)', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_title('Sensitivity to duration of loan', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_xlabel('Duration of loan (years)', fontdict={'size': 14, 'family': 'serif'})

    # Set tick parameters for all plots
    for ax in axes:
        ax.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)

    # Add a single legend for all subplots in the upper left of the figure
    fig.legend([capex_bars, capex_irr_bars], ['NPV (MDKK)', 'IRR (%)'], loc='upper left', ncol=1, fontsize=14, bbox_to_anchor=(0.0, 1))

    # Set layout
    plt.tight_layout()

def plot_sensitivity1_price_wind(path):
    # Load the Excel files

    price_df = pd.read_excel(path + 'sensitivity_price.xlsx', header=None)
    wind_df = pd.read_excel(path + 'sensitivity_wind.xlsx', header=None)

    # Prepare data for plotting
    scenarios = ['LOW', 'MED', 'HIGH']
    probability = ['P50', 'P75', 'P90']

    # Extract data for plotting
    price_npv, price_irr = price_df.iloc[:, 0], price_df.iloc[:, 1]
    wind_npv, wind_irr = wind_df.iloc[:, 0], wind_df.iloc[:, 1]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # Set limits manually to align zeros
    npv_min = min(price_npv.min(), wind_npv.min(), 0) - 500
    npv_max = max(price_npv.max(), wind_npv.max()) + 1000
    irr_min = npv_min / 800  # Escalar IRR para que esté alineado con NPV
    irr_max = npv_max / 800   # Ajuste para que IRR y NPV tengan el mismo "peso visual"

    plt.rc('font', family='serif', size=14)  # Set font style to serif and size to 14

    # First subplot for Price sensitivity with secondary axis
    ax1 = axes[0]
    price_bars = ax1.bar(scenarios, price_npv, label='NPV (MDKK)', color='skyblue', width=0.4)
    ax1.set_ylim(npv_min, npv_max)
    ax1.set_ylabel('NPV (MDKK)', fontdict={'size': 14, 'family': 'serif'})
    ax2 = ax1.twinx()
    price_irr_bars = ax2.bar(scenarios, price_irr, label='IRR (%)', color='salmon', width=0.2, alpha=0.7)
    ax2.set_ylim(irr_min, irr_max)
    ax2.set_ylabel('IRR (%)', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_title('Sensitivity to Price', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_xlabel('Scenarios', fontdict={'size': 14, 'family': 'serif'})

    # Second subplot for Wind sensitivity with secondary axis
    ax1 = axes[1]
    wind_bars = ax1.bar(probability, wind_npv, label='NPV (MDKK)', color='skyblue', width=0.4)
    ax1.set_ylim(npv_min, npv_max)
    ax1.set_ylabel('NPV (MDKK)', fontdict={'size': 14, 'family': 'serif'})
    ax2 = ax1.twinx()
    wind_irr_bars = ax2.bar(probability, wind_irr, label='IRR (%)', color='salmon', width=0.2, alpha=0.7)
    ax2.set_ylim(irr_min, irr_max)
    ax2.set_ylabel('IRR (%)', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_title('Sensitivity to Wind', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_xlabel('Probability', fontdict={'size': 14, 'family': 'serif'})

    # Set tick parameters for all plots
    for ax in axes:
        ax.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)

    # Add a single legend for all subplots in the upper left of the figure
    fig.legend([price_bars, price_irr_bars], ['NPV (MDKK)', 'IRR (%)'], loc='upper left', ncol=1, fontsize=14, bbox_to_anchor=(0, 1))

    # Set layout
    plt.tight_layout()

def plot_sensitivity_HYDROGEN_CFD():
    # Load the Excel files
    path = '../data/sensitivity3/'
    cfd_df = pd.read_excel(path + 'sensitivity_cfd.xlsx', header=None)
    hydrogen_df = pd.read_excel(path + 'sensitivity_hydrogen.xlsx', header=None)

    # Extract data for plotting
    cfd_prices = cfd_df.iloc[:, 0]  # CfD prices (DKK/MWh)
    cfd_npv = cfd_df.iloc[:, 1]   # NPV values for CfD
    print("cfd_npv: ", cfd_npv)
    hydrogen_prices = hydrogen_df.iloc[:, 0]  # Hydrogen prices (EUR/kg)
    hydrogen_npv = hydrogen_df.iloc[:, 1]     # NPV values for hydrogen
    print("hydrogen_npv: ", hydrogen_npv)
    # Set common Y-axis limits
    npv_min = min(cfd_npv.min(), hydrogen_npv.min(), 0) - 500
    npv_max = max(cfd_npv.max(), hydrogen_npv.max()) + 1000

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    plt.rc('font', family='serif', size=14)  # Set font style to serif and size to 14

    # First subplot: NPV vs CfD Price
    ax1 = axes[0]
    ax1.bar(cfd_prices, cfd_npv, label='NPV (MDKK)', color='skyblue', width=20)
    ax1.set_ylim(npv_min, npv_max)
    ax1.set_xlabel('CfD Price (DKK/MWh)', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_ylabel('NPV (MDKK)', fontdict={'size': 14, 'family': 'serif'})
    ax1.set_title('NPV Sensitivity to CfD Price', fontdict={'size': 14, 'family': 'serif'})
    ax1.tick_params(axis='both', labelsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Second subplot: NPV vs Hydrogen Price
    ax2 = axes[1]
    ax2.bar(hydrogen_prices, hydrogen_npv, label='NPV (MDKK)', color='skyblue', width=0.4)
    ax2.set_ylim(npv_min, npv_max)
    ax2.set_xlabel('Hydrogen Price (EUR/kg)', fontdict={'size': 14, 'family': 'serif'})
    ax2.set_ylabel('NPV (MDKK)', fontdict={'size': 14, 'family': 'serif'})
    ax2.set_title('NPV Sensitivity to Hydrogen Price', fontdict={'size': 14, 'family': 'serif'})
    ax2.tick_params(axis='both', labelsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Set layout
    plt.tight_layout()
    



    
if __name__ == '__main__':
    path = '../data/sensitivity2/'
    plot_sensitivity1_CAPEX_OPEX_LOAN(path = path)
    plot_sensitivity1_price_wind(path = path)
    #plot_sensitivity_HYDROGEN_CFD()
    plt.show()