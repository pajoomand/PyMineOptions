import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.io import loadmat
# --- Global Project Parameters ---
project_years = 34 # Defines the total duration of the project in years
first_year = 5 # Specifies a reference year, potentially for calculation start points like depreciation

x = loadmat('data/cashflow_fixed_parameters.mat')
class cash_flow_inputs():
    """
    A container class to aggregate all the necessary input variables and fixed parameters
    that feed into the detailed cash flow calculations.
    """
    def __init__(self, ROM, Yield, Exchange_Rate, Price, OPEX, Transport, Port, Capex, Royalty_fixed, Tax_Rate, Working_capital_days, Marketing_fixed):
        """
        Initialises all cash flow related input variables.

        Args:
            ROM (Input__risk_variable): Run-of-Mine production variable.
            Yield (Input__risk_variable): Material yield variable.
            Exchange_Rate (Input__risk_variable): Currency exchange rate variable.
            Price (Input__risk_variable): Product price variable.
            OPEX (Input__risk_variable): Operating expenses variable.
            Transport (Input__risk_variable): Transport costs variable.
            Port (Input__risk_variable): Port costs variable.
            Capex (Input__risk_variable): Capital expenditures variable.
            Royalty_fixed (float): Fixed royalty rate.
            Tax_Rate (float): Corporate tax rate.
            Working_capital_days (int): Number of days of working capital required.
            Marketing_fixed (float): Fixed marketing cost percentage of revenue.
        """
        self.ROM = ROM
        self.Yield = Yield
        self.Exchange_Rate = Exchange_Rate
        self.Price = Price
        self.OPEX = OPEX
        self.Transport = Transport
        self.Port = Port
        self.Capex = Capex
        self.Royalty_fixed = Royalty_fixed
        self.Tax_Rate = Tax_Rate
        self.Working_capital_days = Working_capital_days
        self.Marketing_fixed = Marketing_fixed

#==========================================================================================================================================

