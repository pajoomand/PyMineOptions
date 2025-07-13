import numpy as np
import math
from scipy.ndimage.interpolation import shift # Imported but not used in this specific GUI logic.
from scipy.io import loadmat # Used for loading .mat files, with fallback for demonstration.
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

# --- Mock Data for .mat file import ---
# In a real application, you would load fixed parameters from 'cashflow_fixed_parameters.mat'.
# This try-except block provides mock data if the file is not found,
# allowing the script to run for demonstration purposes.
try:
    x = loadmat('cashflow_fixed_parameters.mat')
    print("Loaded data from 'cashflow_fixed_parameters.mat'.")
except FileNotFoundError:
    print("Warning: 'cashflow_fixed_parameters.mat' not found. Using mock data for fixed parameters.")
    x = {
        "ROM_fixed_2_3": 0.05,  # Example fixed value for Run-of-Mine
        "Yield_fixed": 0.02,    # Example fixed value for Yield
        "capex_fixed": 100      # Example fixed value for Capital Expenditure
    }

# --- Module 1: Financial Models, Risk Variables, and Cash Flow Simulation ---
# These classes are integral to defining distributions, risk variables,
# and the core cash flow model for project valuation.

class MonteCarloP:
    """
    Simulates option prices using the Monte Carlo method.
    (Included for completeness based on 'module1' content; not directly used in this GUI).
    """
    def __init__(self, current_price, strike_price, risk_free_rate, time_to_maturity, volatility, iterations):
        self.current_price = current_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.time_to_maturity = time_to_maturity
        self.volatility = volatility
        self.iterations = iterations

    def get_call_option_price(self):
        np.random.seed(42) # For reproducibility
        log_returns = np.random.normal((self.risk_free_rate - 0.5 * self.volatility**2) * self.time_to_maturity,
                                       self.volatility * np.sqrt(self.time_to_maturity),
                                       self.iterations)
        simulated_prices = self.current_price * np.exp(log_returns)
        payoffs = np.maximum(simulated_prices - self.strike_price, 0)
        option_price = np.exp(-self.risk_free_rate * self.time_to_maturity) * np.mean(payoffs)
        return option_price

    def get_put_option_price(self):
        np.random.seed(42) # For reproducibility
        log_returns = np.random.normal((self.risk_free_rate - 0.5 * self.volatility**2) * self.time_to_maturity,
                                       self.volatility * np.sqrt(self.time_to_maturity),
                                       self.iterations)
        simulated_prices = self.current_price * np.exp(log_returns)
        payoffs = np.maximum(self.strike_price - simulated_prices, 0)
        option_price = np.exp(-self.risk_free_rate * self.time_to_maturity) * np.mean(payoffs)
        return option_price

class BlackScholes:
    """
    Calculates European option prices using the Black-Scholes model.
    (Included for completeness; requires `scipy.stats.norm` for `norm.cdf` if used).
    """
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self):
        # Requires 'from scipy.stats import norm'
        pass # Placeholder: return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        pass # Placeholder: return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        pass # Placeholder: return self.S * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_price(self):
        pass # Placeholder: return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * norm.cdf(-self.d1())

class Distribution:
    """
    Represents a probability distribution for a risk variable.
    Attributes:
        mean (float): The mean (or central tendency) of the distribution.
        std_dev (float): The standard deviation (or spread) of the distribution.
        dist_type (str): The type of distribution (e.g., "Triangular", "Gaussian").
    """
    def __init__(self, mean, std_dev, dist_type):
        self.mean = mean
        self.std_dev = std_dev
        self.dist_type = dist_type

class Input__risk_variable:
    """
    Base class for an input risk variable, associating it with its distribution
    and specific fixed or scenario values.
    Attributes:
        name (str): The descriptive name of the variable (e.g., "ROM", "Price").
        distribution (Distribution): The statistical distribution object.
        fixed_value (float): The base or fixed value of the variable.
        scenario_value (float): A value used in specific scenario analyses (default 0).
        index (int): An arbitrary index for identification (default 0 or 1).
    """
    def __init__(self, name, distribution, fixed_value, scenario_value, index):
        self.name = name
        self.distribution = distribution
        self.fixed_value = fixed_value
        self.scenario_value = scenario_value
        self.index = index

class input_risk_variable_with_spot_percentage(Input__risk_variable):
    """
    Extends Input__risk_variable for variables that have a 'spot percentage' component,
    often relevant in models where a part of the variable is tied to a spot market price.
    Attributes:
        spot_percentage (float): The percentage tied to the spot component (default 1).
    """
    def __init__(self, name, distribution, fixed_value, scenario_value, index, spot_percentage):
        super().__init__(name, distribution, fixed_value, scenario_value, index)
        self.spot_percentage = spot_percentage

class cash_flow_inputs:
    """
    A comprehensive container class to bundle all individual risk variables
    and fixed financial parameters that feed into the cash flow calculation model.
    This simplifies passing inputs to the main cash flow model.
    """
    def __init__(self, ROM, Yield, Exchange_Rate, Price, Opex, Transport, Port, Capex,
                 Royalty_fixed, Tax_Rate, Working_capital_days, Marketing_fixed):
        self.ROM = ROM
        self.Yield = Yield
        self.Exchange_Rate = Exchange_Rate
        self.Price = Price
        self.Opex = Opex
        self.Transport = Transport
        self.Port = Port
        self.Capex = Capex
        self.Royalty_fixed = Royalty_fixed
        self.Tax_Rate = Tax_Rate
        self.Working_capital_days = Working_capital_days
        self.Marketing_fixed = Marketing_fixed

class Fund_info:
    """
    Stores all pertinent financial funding details for a project,
    including equity/debt percentages, amounts, interest rates,
    discount rates, and maturity periods.
    """
    def __init__(self, fund_percentage, fund_amount, interest_rate, nominal_discount_rate, maturity):
        self.fund_percentage = fund_percentage
        self.fund_amount = fund_amount
        self.interest_rate = interest_rate
        self.nominal_discount_rate = nominal_discount_rate
        self.maturity = maturity

class Prototype_Cash_flow: # Renamed from Belvedare_Cash_flow for consistency
    """
    Represents the core Prototype cash flow model. This class is designed to
    encapsulate the financial logic for calculating project cash flows over time.
    It takes various inputs to build a comprehensive financial projection.
    """
    def __init__(self, inputs, number_of_years, fund_info, other_specific_param):
        """
        Initializes the Prototype cash flow model.
        Args:
            inputs (cash_flow_inputs): An instance of the cash_flow_inputs class containing all variable and fixed inputs.
            number_of_years (int): The duration of the project or analysis period in years.
            fund_info (Fund_info): An instance of Fund_info containing funding details.
            other_specific_param: A placeholder for any other specific model parameters.
        """
        self.inputs = inputs
        self.number_of_years = number_of_years 
        self.fund_info = fund_info
        self.other_specific_param = other_specific_param 

    def calculate_NPV(self):
        """
        Placeholder method: In a full implementation, this method would execute
        detailed financial calculations to determine the Net Present Value (NPV)
        of the project based on all its inputs.
        """
        print(f"Calculating NPV for Prototype Cash Flow model using inputs: {self.inputs.ROM.name}, {self.inputs.Yield.name}...")
        return 500.0 # Returns a dummy value for demonstration

    def generate_cash_flows(self):
        """
        Placeholder method: In a full implementation, this would simulate or
        calculate a series of annual (or periodic) cash flows for the project.
        """
        print("Generating dummy cash flows...")
        return np.array([100, 110, 120, 130, 140]) # Returns a simple array for demonstration

class generate_output_sigma:
    """
    A utility class to perform Monte Carlo simulations on a cash flow model.
    Its primary purpose is to estimate the volatility (sigma) of a chosen output
    (e.g., project NPV) by varying specified input risk variables.
    """
    def __init__(self, cash_flow_model, input_variable_names, iterations):
        """
        Initializes the output sigma generator.
        Args:
            cash_flow_model (Prototype_Cash_flow): The instantiated cash flow model to simulate.
            input_variable_names (list of str): A list of names of the input risk variables
                                               (e.g., "Yield", "ROM") whose impact on output sigma is to be studied.
            iterations (int): The number of Monte Carlo simulation runs.
        """
        self.cash_flow_model = cash_flow_model
        self.input_variable_names = input_variable_names
        self.iterations = iterations

    def MC_simulation(self):
        """
        Executes the Monte Carlo simulation.
        In a complete implementation, this method would:
        1. Iterate 'self.iterations' times.
        2. In each iteration, it would sample values for the 'input_variable_names'
           from their respective distributions (found within `self.cash_flow_model.inputs`).
        3. These sampled values would be fed back into the cash flow model.
        4. A key output (e.g., NPV, a specific cash flow) would be calculated.
        5. All collected output values would then be used to compute their standard deviation.

        For this demonstration, predefined plausible sigma values are returned
        based on the selected input variable to enable script execution.
        """
        print(f"Running MC simulation for {self.input_variable_names} to estimate output sigma with {self.iterations} iterations...")
        
        # --- Placeholder for actual Monte Carlo simulation logic ---
        # Returns dummy sigma values based on common patterns for different variables
        if "Yield" in self.input_variable_names:
            return 0.113995 
        elif "ROM" in self.input_variable_names:
            return 0.084794
        elif "Exchange_Rate" in self.input_variable_names:
            return 0.05
        elif "Price" in self.input_variable_names:
            return 0.25
        else:
            return 0.1 # Generic sigma if selected variable not specifically handled

# --- Module 2: Lattice Models for Real Options ---
# These classes are designed for valuing real options using binomial lattice methods.

class RecursiveLattice:
    """
    Base class for constructing and valuing options using a recursive binomial lattice.
    Provides fundamental methods for lattice structure and discounting.
    """
    def __init__(self, root, Maximisation, Rf, SubSteps):
        """
        Initializes the base lattice.
        Args:
            root (float): The initial value of the underlying asset/project at the start of the lattice.
            Maximisation (Maximisation_class): An object containing option-specific parameters (volatilities, costs).
            Rf (float): The risk-free interest rate.
            SubSteps (int): The number of sub-steps within each main time period of the lattice.
        """
        self.root = root
        self.Maximisation = Maximisation
        self.Rf = Rf
        self.SubSteps = SubSteps

    @staticmethod
    def LatticeNodes(L):
        """Calculates the total number of nodes up to a given level L in the lattice."""
        return int((L + 1) * (L + 2) / 2)

    @staticmethod
    def discount_factor(rf, delta_t):
        """Calculates the present value discount factor for a given rate and time step."""
        return (math.exp(-rf * delta_t))

class SimpleRecursiveLattice(RecursiveLattice):
    """
    Implements a single-asset, recursive binomial lattice for valuing a simple real option.
    It calculates the Expanded Net Present Value (ENPV) by working backward
    from the terminal nodes, considering the option's exercise decision.
    """
    def __init__(self, root, Maximisation, Rf, SubSteps):
        super().__init__(root, Maximisation, Rf, SubSteps)

    def Calculate_ENPV(self, current_root, Current_L):
        """
        Recursively calculates the Expanded Net Present Value (ENPV) for a simple option.
        This function builds the lattice forward and then performs backward induction.
        Args:
            current_root (float): The value of the underlying asset at the current node.
            Current_L (int): The current volatility level index to use from Maximisation.STD.
        Returns:
            float: The ENPV at the current node.
        """
        sigma = self.Maximisation.STD[Current_L]
        # Calculate time step (dt) based on total time, number of STD levels, and sub-steps.
        dt = self.Maximisation.T[0] / (len(self.Maximisation.STD) * self.SubSteps)
        U = math.exp(sigma * dt**0.5) # Up factor for the binomial tree
        D = 1 / U                     # Down factor
        p = (math.exp(self.Rf * dt) - D) / (U - D) # Risk-neutral probability of an up move

        ForwardLattice = np.zeros(self.LatticeNodes(self.SubSteps) + 1)
        Pn = current_root # Current node value
        ForwardLattice[1] = Pn # Initialize the root of the current sub-lattice

        # --- Forward Pass: Build the lattice values ---
        for i in range(1, self.SubSteps + 1):
            ForwardLattice[self.LatticeNodes(i)] = Pn * D # Smallest value at this level (all down moves from previous)
            Pn = ForwardLattice[self.LatticeNodes(i)]
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i)):
                ForwardLattice[j] = ForwardLattice[j - i] * U # Calculate intermediate nodes by moving up from prior level

        # --- Backward Pass: Value the option at each node ---
        Total_levels = len(self.Maximisation.STD) - 1

        if Current_L < Total_levels:
            # If not at the final volatility level, recursively call for the next level
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i) + 1):
                ForwardLattice[j] = self.Calculate_ENPV(ForwardLattice[j], Current_L + 1)
        else:
            # At the last volatility level (terminal nodes of the overall lattice), apply exercise decision
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i) + 1):
                # Calculate the value if the option is exercised: current project value minus discounted investment cost
                exercise_value = ForwardLattice[j] - (self.Maximisation.investments[0] / math.exp(self.Maximisation.NomDiscountFactor * self.Maximisation.T[0]))
                result = max(exercise_value, 0) # The option value is the maximum of exercising or letting it expire worthless
                ForwardLattice[j] = result

        # Work backwards through the lattice, discounting and combining expected values
        for i in range(self.SubSteps, 0, -1):
            for j in range((self.LatticeNodes(i - 1) - i + 1), self.LatticeNodes(i - 1) + 1):
                # Calculate the expected value at each node by weighting up and down moves by risk-neutral probabilities
                ForwardLattice[j] = ((ForwardLattice[j + i] * p) + (ForwardLattice[j + i + 1] * (1 - p)))
                # Discount back to the current node, unless it's the very first node of the entire tree
                if not (j == 1 and Current_L == 0):
                    ForwardLattice[j] = ForwardLattice[j] * self.discount_factor(self.Rf, dt)

        return ForwardLattice[1] # Return the final ENPV at the initial root node

class CompoundRecursiveLattice(RecursiveLattice):
    """
    Implements a lattice model for valuing compound real options, which involve
    multiple, sequential investment decisions or stages.
    """
    def __init__(self, root, Maximisation, Rf, SubSteps):
        super().__init__(root, Maximisation, Rf, SubSteps)

    @staticmethod
    def NominalDiscountFactor(endpoint, maximisation, c):
        """
        Calculates the nominal discounted value for a specific stage of a compound option.
        This incorporates the equity percentage and investment costs for that stage.
        """
        return (endpoint * maximisation.Percentage_Equity[c] - (maximisation.investments[c]))

    def Calculate_ENPV(self, current_root, Current_L):
        """
        Recursively calculates the Expanded Net Present Value (ENPV) for a compound option.
        This method extends the simple lattice to handle multiple decision points and their values.
        """
        sigma = self.Maximisation.STD + self.Maximisation.STD # Volatility handling for compound options
        maxi_compound_level = len(self.Maximisation.STD) - 1
        dt = self.Maximisation.T[0] / (len(self.Maximisation.STD) * self.SubSteps)

        U = math.exp(sigma[Current_L] * dt**0.5)
        D = 1 / U
        p = (math.exp(self.Rf * dt) - D) / (U - D)

        ForwardLattice = np.zeros(self.LatticeNodes(self.SubSteps) + 1)
        forward_maximisation = np.zeros(self.LatticeNodes(self.SubSteps) + 1)
        Pn = current_root
        ForwardLattice[1] = Pn

        Total_levels = len(sigma) - 1

        for i in range(1, self.SubSteps + 1):
            ForwardLattice[self.LatticeNodes(i)] = Pn * D
            Pn = ForwardLattice[self.LatticeNodes(i)]
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i)):
                ForwardLattice[j] = ForwardLattice[j - i] * U

            if Current_L == maxi_compound_level:
                forward_maximisation[self.LatticeNodes(i-1)+1 : self.LatticeNodes(i)+1] = ForwardLattice[self.LatticeNodes(i-1)+1 : self.LatticeNodes(i)+1]
        
        if Current_L < Total_levels:
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i) + 1):
                ForwardLattice[j] = self.Calculate_ENPV(ForwardLattice[j], Current_L + 1)

            if Current_L == maxi_compound_level:
                for v in range(self.LatticeNodes(i - 1) + 1, self.LatticeNodes(i) + 1):
                    compound_exercise_val = self.NominalDiscountFactor(forward_maximisation[v], self.Maximisation, 0)
                    ForwardLattice[v] = max(compound_exercise_val, (max(ForwardLattice[v] - self.Maximisation.investments[0], 0)))
        else:
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i) + 1):
                temp = self.NominalDiscountFactor(ForwardLattice[j], self.Maximisation, 1)
                result = max(temp, 0)
                ForwardLattice[j] = result

        for i in range(self.SubSteps, 0, -1):
            for j in range((self.LatticeNodes(i - 1) - i + 1), self.LatticeNodes(i - 1) + 1):
                ForwardLattice[j] = ((ForwardLattice[j + i] * p) + (ForwardLattice[j + i + 1] * (1 - p)))
                if not (j == 1 and Current_L == 0):
                    ForwardLattice[j] = ForwardLattice[j] * self.discount_factor(self.Rf, dt)

        return ForwardLattice[1]

class Maximisation_class():
    """
    Manages parameters specific to the option's exercise and maximisation stages.
    This includes equity contributions, expenses, nominal discount factors,
    volatilities (STD), and the timing of investment decisions (Year).
    """
    def __init__(self, equity, Percentage_Equity, expenses, NomDiscountFactor, STD, Year):
        """
        Initializes Maximisation_class.
        Args:
            equity (float): Total initial equity contribution.
            Percentage_Equity (list): List of equity percentages for each stage/decision point.
            expenses (list): List of expenses for each stage.
            NomDiscountFactor (float): Nominal discount factor applied to investments.
            STD (list): List of standard deviations (volatilities) for different levels/stages.
            Year (list): List of years when investment decisions occur.
        """
        self.equity = equity
        self.expenses = expenses
        self.NomDiscountFactor = NomDiscountFactor
        self.STD = STD
        self.Percentage_Equity = Percentage_Equity
        self.Year = Year

        # Calculate time differences between investment decision years
        t = [0] * len(self.Year)
        if len(self.Year) > 0:
            t[0] = self.Year[0]
            if (len(self.Year) > 1):
                for i in range(1, len(self.Year)):
                    t[i] = self.Year[i] - self.Year[i - 1]
        self.T = t # Store time differences

        # Calculate discounted investment amounts for each stage
        invest = [0] * len(self.Year)
        increased_equity = 0 # Tracks cumulative equity percentage

        for i in range(len(self.Year)):
            # Calculate the additional equity percentage needed for the current stage
            additional_percentage_for_stage = self.Percentage_Equity[i] - increased_equity
            # Calculate the nominal investment for this stage (equity portion + explicit expenses)
            nominal_investment = additional_percentage_for_stage * self.equity + self.expenses[i]
            
            # Discount the nominal investment back to present value using the nominal discount factor
            invest[i] = nominal_investment / math.exp(self.NomDiscountFactor * self.Year[i])
            
            # Update the cumulative equity percentage
            increased_equity = self.Percentage_Equity[i] + increased_equity 
        self.investments = invest # Store the discounted investment amounts

# --- End Module 2 ---

# --- Function to Create Prototype Cash Flow Instance ---
# This function encapsulates the creation and configuration of input variables and the cash flow model.
def create_Prototype_Cash_flow():
    """
    Initializes and configures all necessary input risk variables and fixed parameters
    for the Prototype Cash Flow model. It then returns a fully set-up instance of the
    Prototype_Cash_flow class, ready for simulation or valuation.
    """
    # --- Define Distributions for Input Risk Variables ---
    # These distributions characterize the uncertainty associated with key financial variables.
    ROM_distribution = Distribution(0, .076, "Triangular")
    Yield_distribution = Distribution(0, .035, "Triangular")
    Price_distribution = Distribution(55.7674, .1 * 55.7674, "Gaussian")
    Exchange_Rate_distribution = Distribution(0.7, .07690 * .7, "Gaussian")
    Opex_distribution = Distribution(0, .1, "Triangular")
    Transport_distribution = Distribution(0, .1, "Triangular")
    Port_distribution = Distribution(0, .1, "Triangular")
    Capex_distribution = Distribution(0, .1, "Triangular")

    # --- Instantiate Input Risk Variables with their Distributions and Fixed Values ---
    # Each variable is created with its specific distribution and a base (fixed) value.
    ROM_fixed = x["ROM_fixed_2_3"]
    ROM = input_risk_variable_with_spot_percentage("ROM", ROM_distribution, ROM_fixed, 0, 0, 1)

    Yield_fixed = x["Yield_fixed"]
    Yield = Input__risk_variable("Yield", Yield_distribution, Yield_fixed, 0, 1)

    Price_fixed = Price_distribution.mean
    Price = Input__risk_variable("Price", Price_distribution, Price_fixed, 0, 1)

    Exchange_Rate_fixed = Exchange_Rate_distribution.mean
    Exchange_Rate = Input__risk_variable("Exchange_Rate", Exchange_Rate_distribution, Exchange_Rate_fixed, 0, 1)

    Opex_fixed = Opex_distribution.mean
    Opex = Input__risk_variable("Opex", Opex_distribution, Opex_fixed, 0, 1)

    Transport_fixed = 5.15
    Transport = Input__risk_variable("Transport", Transport_distribution, Transport_fixed, 0, 1)

    Port_fixed = 3.28
    Port = Input__risk_variable("Port", Port_distribution, Port_fixed, 0, 1)

    Capex_fixed = x["capex_fixed"]
    Capex = Input__risk_variable("Capex", Capex_distribution, Capex_fixed, 0, 1)

    # --- Define Fixed Cash Flow Parameters ---
    # These are constant parameters used in the cash flow calculations.
    Royalty_fixed = 0.07
    Tax_Rate = 0.3
    Working_capital_days = 30
    Marketing_fixed = 0.0125

    # --- Aggregate Inputs and Funding Info, then Instantiate Cash Flow Model ---
    inputs_to_Prototype = cash_flow_inputs( # Renamed variable for consistency
        ROM, Yield, Exchange_Rate, Price, Opex, Transport, Port, Capex,
        Royalty_fixed, Tax_Rate, Working_capital_days, Marketing_fixed
    )

    prototype_fund_info = Fund_info( # Renamed variable for consistency
        fund_percentage=1, # E.g., 100% equity for initial setup
        fund_amount=445000, # Total initial project funding
        interest_rate=0.1,  # Example interest rate
        nominal_discount_rate=0.15, # Example nominal discount rate
        maturity=30         # Example project/funding maturity period
    )
    
    # Instantiate the main Prototype Cash Flow model with all prepared inputs.
    # '34' and '0' are example placeholder parameters for the model's internal logic.
    prototype_cashflow_model = Prototype_Cash_flow(inputs_to_Prototype, 34, prototype_fund_info, 0)
    return prototype_cashflow_model


# --- Global Initialization of Financial Model Components ---
# These components are initialized once when the script starts.
# They are used by the GUI to perform calculations.

# Create a single instance of the Prototype Cash Flow model.
# This model contains all the necessary risk variables and fixed parameters.
prototype_cashflow_model = create_Prototype_Cash_flow()

# Global list to keep track of risk variables currently selected by the user via checkboxes.
selected_var = []

# --- GUI Application Class: Mining Valuation & Real Options (PyMineOptions) ---

class PyMineOptionsApp:
    """
    Main application class for the Mining Valuation & Real Options (PyMineOptions) GUI.
    This class encapsulates the Tkinter window, all its widgets, and the
    event-handling logic for user interactions.
    """
    def __init__(self, master):
        """
        Initializes the PyMineOptions GUI application.
        Args:
            master (Tk): The root Tkinter window object.
        """
        self.master = master
        master.title("Mining Valuation & Real Options (PyMineOptions)")
        master.minsize(600, 400) # Set a minimum size for the main window

        # --- Tkinter Control Variables ---
        # These variables are used to link widget states (like text, checkbox status)
        # to Python variables, allowing for dynamic updates.
        self.output_text_var = StringVar() # Manages text displayed in the output label
        self.output_text_var.set("")      # Initialize output label as empty
        
        self.rom_check_var = IntVar()       # Stores 0 or 1 for ROM checkbox state
        self.yield_check_var = IntVar()     # Stores 0 or 1 for Yield checkbox state
        self.exchange_rate_check_var = IntVar() # Stores 0 or 1 for Exchange Rate checkbox state
        self.price_check_var = IntVar()     # Stores 0 or 1 for Price checkbox state
        
        self.iterations_var = StringVar()   # Stores the selected number of iterations from the Combobox

        # --- GUI Widgets Setup ---
        # Widgets are created and placed within the main window using the .grid() layout manager.

        # Output Label: Displays messages, errors, and simulation results to the user.
        self.output_label = Label(master, textvariable=self.output_text_var, wraplength=550, justify=LEFT)
        self.output_label.grid(row=9, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

        # Simulate Button: Triggers the Monte Carlo simulation when clicked.
        self.simulate_button = Button(master, text="Simulate", command=self.simulate_button_tapped)
        self.simulate_button.grid(row=6, column=1, padx=20, pady=20)

        # Iterations Input: Label and Combobox for selecting the number of simulation runs.
        self.iteration_label = Label(master, text='Number of iterations:')
        self.iteration_label.grid(row=0, column=0, padx=20, pady=20, sticky=W)

        self.iterations_combobox = ttk.Combobox(master, width=15, textvariable=self.iterations_var)
        self.iterations_combobox['values'] = (500, 1000, 2000, 3000, 5000, 10000, 30000) # Predefined iteration options
        self.iterations_combobox.grid(column=1, row=0, padx=20, sticky=W)
        self.iterations_combobox.current(0) # Set the default selected value to the first in the list

        # Risk Variable Checkboxes: Allow users to select which input variables to include in the simulation.
        # Each checkbox's command is linked to 'self.select_variables' to dynamically update the selected list.
        self.rom_check = Checkbutton(master, text='ROM', command=self.select_variables, variable=self.rom_check_var)
        self.rom_check.grid(row=2, column=1, padx=20, pady=5, sticky=W)

        self.yield_check = Checkbutton(master, text='Yield', command=self.select_variables, variable=self.yield_check_var)
        self.yield_check.grid(row=3, column=1, padx=20, pady=5, sticky=W)

        self.exchange_rate_check = Checkbutton(master, text='Exchange Rate', command=self.select_variables, variable=self.exchange_rate_check_var)
        self.exchange_rate_check.grid(row=4, column=1, padx=20, pady=5, sticky=W)

        self.price_check = Checkbutton(master, text='Price', command=self.select_variables, variable=self.price_check_var)
        self.price_check.grid(row=5, column=1, padx=20, pady=5, sticky=W)

        # Configure grid columns to expand proportionally when the window is resized.
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)

    def select_variables(self):
        """
        This method is called whenever any of the risk variable checkboxes are clicked.
        It updates the global 'selected_var' list to reflect which variables are currently checked.
        """
        global selected_var # Access the global list that holds selected variable names
        temp_selected_vars = [] # Create a temporary list to build the new selection
        
        # Check the state of each IntVar associated with a checkbox.
        # If checked (value is 1), add the corresponding variable name to the temporary list.
        if self.rom_check_var.get():
            temp_selected_vars.append("ROM")
        if self.yield_check_var.get():
            temp_selected_vars.append("Yield")
        if self.exchange_rate_check_var.get():
            temp_selected_vars.append("Exchange_Rate")
        if self.price_check_var.get():
            temp_selected_vars.append("Price")
        
        selected_var = temp_selected_vars # Update the global list with the new selection
        print(f"Variables selected for simulation: {selected_var}") # Print to console for debugging/tracking

    def simulate_button_tapped(self):
        """
        This method is executed when the 'Simulate' button is pressed.
        It orchestrates the simulation process:
        1. Validates user input (ensures at least one variable is selected).
        2. Retrieves the selected number of iterations.
        3. Calls the Monte Carlo simulation method from `generate_output_sigma`.
        4. Displays the simulation result or an error message in the GUI.
        """
        global prototype_cashflow_model, selected_var # Access the global model instance and selected variables
        
        # Input Validation: Ensure at least one variable is selected before proceeding.
        if not selected_var:
            self.output_text_var.set("Error: Please select at least one variable for simulation.")
            messagebox.showwarning("Selection Error", "Please select at least one variable to run the simulation.")
            print("Simulation aborted: No variables selected.")
            return

        try:
            # Safely convert the iterations string from the combobox to an integer.
            num_iterations = int(self.iterations_var.get())
            
            # Create an instance of the simulator with the cash flow model, selected variables, and iterations.
            output_sigma_calculator = generate_output_sigma(prototype_cashflow_model, selected_var, num_iterations)
            
            # Execute the Monte Carlo simulation and round the result for display.
            calculated_sigma = output_sigma_calculator.MC_simulation().__round__(3)
            
            # Update the GUI's output label with the simulation result.
            self.output_text_var.set(f"Simulation Complete!\nOutput sigma for {', '.join(selected_var)}: {calculated_sigma}")
            messagebox.showinfo("Simulation Result", f"The calculated output sigma is: {calculated_sigma}")
            print(f"Simulation successful. Output sigma for {selected_var}: {calculated_sigma}")
            
        except ValueError:
            # Handle cases where the number of iterations might not be a valid integer.
            self.output_text_var.set("Error: Invalid number of iterations selected. Please choose a value from the dropdown.")
            messagebox.showerror("Input Error", "Please select a valid number of iterations from the dropdown list.")
            print("Simulation error: Invalid iterations input.")
        except Exception as e:
            # Catch any other unexpected errors during the simulation process.
            self.output_text_var.set(f"An unexpected error occurred during simulation: {e}")
            messagebox.showerror("Simulation Error", f"An unexpected error occurred: {e}")
            print(f"An unexpected error occurred during simulation: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Create the main Tkinter root window.
    root = Tk()
    
    # Instantiate the PyMineOptions, passing the root window.
    # This sets up the entire GUI.
    app = PyMineOptionsApp(root)
    
    # Start the Tkinter event loop.
    # This keeps the GUI window open and responsive to user interactions
    # until the window is closed.
    root.mainloop()