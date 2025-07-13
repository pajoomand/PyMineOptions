from scipy.ndimage.interpolation import shift
from scipy.io import loadmat
from module1 import * # Importing all classes and functions from your module1.py
def create_Prototype_Cash_flow():
    """
    This function encapsulates the creation and instantiation of various
    risk variables, fixed parameters, and the Prototype Cash Flow model.
    It prepares all necessary inputs for the cash flow simulation or valuation.
    """
    # --- Define Distributions for Input Risk Variables ---
    # These define the statistical properties (mean, standard deviation, type)
    # for each variable that introduces uncertainty into the cash flow model.

    # ROM (Run-of-Mine) distribution: Triangular distribution with mean 0 and std_dev 0.076.
    # Triangular distributions are often used to model variations around a base value.
    ROM_distribution = Distribution(0, .076, "Triangular")
    # Yield distribution: Triangular distribution with mean 0 and std_dev 0.035.
    Yield_distribution = Distribution(0, .035, "Triangular")
    # Price distribution: Gaussian (Normal) distribution. Mean is 55.7674, std_dev is 10% of the mean.
    Price_distribution = Distribution(55.7674, .1 * 55.7674, "Gaussian")
    # Exchange Rate distribution: Gaussian (Normal) distribution. Mean is 0.7, std_dev is 0.07690 * 0.7.
    Exchange_Rate_distribution = Distribution(0.7, .07690 * .7, "Gaussian")
    # Opex (Operating Expenditure) distribution: Triangular distribution for variation. Mean 0, std_dev 0.1.
    Opex_distribution = Distribution(0, .1, "Triangular")
    # Transport cost distribution: Triangular distribution for variation. Mean 0, std_dev 0.1.
    Transport_distribution = Distribution(0, .1, "Triangular")
    # Port cost distribution: Triangular distribution for variation. Mean 0, std_dev 0.1.
    Port_distribution = Distribution(0, .1, "Triangular")
    # Capex (Capital Expenditure) distribution: Triangular distribution for variation. Mean 0, std_dev 0.1.
    Capex_distribution = Distribution(0, .1, "Triangular")

    # --- Instantiate Input Risk Variables with their Distributions and Fixed Values ---
    # Each risk variable is instantiated with its defined distribution and a fixed (or base) value.
    # Some fixed values are loaded from the 'x' dictionary (simulating .mat file import),
    # others are derived from their distributions' means or hardcoded.

    # ROM variable: Uses input_risk_variable_with_spot_percentage because it likely has a spot component.
    # ROM_fixed_2_3 is loaded from the 'x' dictionary.
    ROM_fixed = x["ROM_fixed_2_3"] # Assumes 'ROM_fixed_2_3' exists in the loaded .mat file data
    ROM = input_risk_variable_with_spot_percentage("ROM", ROM_distribution, ROM_fixed, 0, 0, 1)

    # Yield variable: Uses Input__risk_variable. Yield_fixed is loaded from 'x'.
    Yield_fixed = x["Yield_fixed"] # Assumes 'Yield_fixed' exists in the loaded .mat file data
    Yield = Input__risk_variable("Yield", Yield_distribution, Yield_fixed, 0, 1)

    # Price variable: Uses Input__risk_variable. Price_fixed is set to the mean of its distribution.
    Price_fixed = Price_distribution.mean
    Price = Input__risk_variable("Price", Price_distribution, Price_fixed, 0, 1)

    # Exchange Rate variable: Uses Input__risk_variable. Exchange_Rate_fixed is set to the mean of its distribution.
    Exchange_Rate_fixed = Exchange_Rate_distribution.mean
    Exchange_Rate = Input__risk_variable("Exchange_Rate", Exchange_Rate_distribution, Exchange_Rate_fixed, 0, 1)

    # OPEX variable: Uses Input__risk_variable. Opex_fixed is set to the mean of its distribution (0 for triangular variation).
    Opex_fixed = Opex_distribution.mean
    Opex = Input__risk_variable("Opex", Opex_distribution, Opex_fixed, 0, 1)

    # Transport variable: Uses Input__risk_variable. Transport_fixed is a hardcoded value.
    Transport_fixed = 5.15
    Transport = Input__risk_variable("Transport", Transport_distribution, Transport_fixed, 0, 1)

    # Port variable: Uses Input__risk_variable. Port_fixed is a hardcoded value.
    Port_fixed = 3.28
    Port = Input__risk_variable("Port", Port_distribution, Port_fixed, 0, 1)

    # Capex variable: Uses Input__risk_variable. Capex_fixed is loaded from 'x'.
    Capex_fixed = x["capex_fixed"] # Assumes 'capex_fixed' exists in the loaded .mat file data
    Capex = Input__risk_variable("Capex", Capex_distribution, Capex_fixed, 0, 1)

    # --- Define Fixed Cash Flow Parameters ---
    # These are constant values used in the cash flow calculations that are not subject to distributions.
    Royalty_fixed = 0.07        # Fixed royalty rate
    Tax_Rate = 0.3              # Tax rate
    Working_capital_days = 30   # Number of days for working capital
    Marketing_fixed = 0.0125    # Fixed marketing cost/rate

    # --- Instantiate Cash Flow Inputs, Funding Info, and Cash Flow Model ---

    # Consolidate all instantiated risk variables and fixed parameters into a single object
    # for the cash flow model.
    inputs_to_Prototype = cash_flow_inputs(
        ROM, Yield, Exchange_Rate, Price, Opex, Transport, Port, Capex,
        Royalty_fixed, Tax_Rate, Working_capital_days, Marketing_fixed
    )

    # Define funding information for the project.
    Prototype_fund_info = Fund_info(
        fund_percentage=1, # 100% funding percentage (e.g., all equity or all debt for initial setup)
        fund_amount=445000, # Total fund amount for the project
        interest_rate=0.1,  # Interest rate on borrowed funds (if any)
        nominal_discount_rate=0.15, # Nominal discount rate for future cash flows
        maturity=30         # Maturity period of funding (e.g., loan term in years)
    )

    # Instantiate the Prototype_Cash_flow model itself.
    # It takes the aggregated inputs, a numeric parameter (e.g., number of years for analysis),
    # the funding information, and another generic parameter (0 in this case).
    cashflow_model=Prototype_Cash_flow(inputs_to_Prototype, 34, Prototype_fund_info, 0)
    
    # Return the fully configured cash flow model instance.
    return cashflow_model

# Example of how you might use the function:
# Prototype_cf = create_Prototype_Cash_flow()
