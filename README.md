## OptimisingWindFarm
A Python program that optimizes wind farm capacity based on the number of turbines (N). It maximizes the net present value (NPV) of the project, while enforcing constraints on the installed capacity. Includes command-line input for easy adjustment and validation of parameters.

## Project Structure
```
optimisingWindFarm/
│
├── data/
│   ├── data.txt                             # General project data (inflation, taxes, loan details, etc.)
│   ├── electricity_price_forecast.csv       # Forecast of electricity prices over time
│   ├── power_output.csv                     # Power output data from the wind farm
│   ├── sensitivity1/                        # Folder with sensitivity analysis data (scenario 1)
│   ├── sensitivity2/                        # Folder with sensitivity analysis data (scenario 2)
│   └── sensitivity3/                        # Folder with sensitivity analysis data (scenario 3)
│
├── scripts/
│   ├── load_data_functions.py               # Functions for loading and processing data
│   ├── calculations_functions.py            # Functions for performing financial calculations like NPV
│   ├── plot_functions.py                    # Functions for visualizing data and results
│   └── plots.py                             # Script for generating plots using the defined functions
│
├── figures/                                 # Folder to store generated plots
│
├── main.py                                  # Main script to run the optimization process
│
└── README.md                                # Project documentation file


```
## How to run
1. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib numpy-financial
   ```
3. Run the main script:
   ```bash
   python main.py
   ```
## Notes
- Ensure data.txt, electricity_price_forecast.csv, and power_output.csv are placed in the data/ directory before running the script.
- The script optimizes the installed capacity of a wind farm and maximizes the net present value (NPV) of the project, based on provided project data, loan details, and electricity price forecasts.
