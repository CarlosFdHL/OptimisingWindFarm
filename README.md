# OptimisingWindFarm
A Python program that optimizes wind farm capacity based on the number of turbines (N). It maximizes the net present value (NPV) of the project, while enforcing constraints on the installed capacity. Includes command-line input for easy adjustment and validation of parameters.

Project/
│
├── data/
│   ├── data.txt                             # General project data (inflation, taxes, loan details, etc.)
│   ├── electricity_price_forecast.csv       # Forecast of electricity prices over time
│   └── power_output.csv                     # Power output data for the wind farm
├── scripts/
│   ├── load_data_functions.py               # Functions to load and process data
│   ├── calculations_functions.py            # Functions to perform NPV and other financial calculations
│   └── plot_functions.py                    # Functions to visualize the results and data
└── main.py                                  # Main script to run the optimization process

# How to run
1. Install dependencies:
   pip install numpy scipy matplotlib numpy-financial
3. Run the main script:
   python main.py

#Notes
- Ensure data.txt, electricity_price_forecast.csv, and power_output.csv are placed in the data/ directory before running the script.
- The script optimizes the installed capacity of a wind farm and maximizes the net present value (NPV) of the project, based on provided project data, loan details, and electricity price forecasts.
