# This line imports all classes and functions from 'module2' and 'Module 1.
from module2 import *
from create_Prototype_Cash_flow import *
from module1 import * # Initialize an instance of Maximisation_class to define parameters for the simple option.
# --- Main Script: Demonstrating Framework Usage ---

# 1. Create an instance of the Prototype Cash Flow model using the helper function.
# This encapsulates all the input variables and fixed parameters.
Prototype = create_Prototype_Cash_flow()

# 2. Calculate output volatilities (sigmas) for specific input variables
#    using Monte Carlo simulation.
#    - For 'Yield': Simulates the Prototype model varying 'Yield' to estimate its impact's standard deviation.
#    - For 'ROM': Simulates the Prototype model varying 'ROM' to estimate its impact's standard deviation.
#    The results are rounded to 3 decimal places.
Yield_output_sigma = generate_output_sigma(Prototype, ["Yield"], 5000).MC_simulation().__round__(3)
ROM_output_sigma = generate_output_sigma(Prototype, ["ROM"], 5000).MC_simulation().__round__(3)

# 3. Initialize an instance of Maximisation_class with calculated sigmas and other parameters.
#    This class defines the financial parameters for the real option valuation.
#    - equity: Initial equity contribution.
#    - Percentage_Equity: Equity percentages for different stages.
#    - expenses: Expenses incurred at different stages.
#    - NomDiscountFactor: Nominal discount factor.
#    - STD: This is crucial – it now uses the dynamically calculated `Yield_output_sigma`
#           and `ROM_output_sigma` as the standard deviations for the lattice model.
#    - Year: The year(s) when investment decisions can be made.
Prototype_simple_option=Maximisation_class(equity=128.5714+123.529,
                                           Percentage_Equity=[1],
                                           expenses=[16.6],
                                           NomDiscountFactor=.15,
                                           STD=[Yield_output_sigma, ROM_output_sigma], # Using the calculated sigmas here
                                           Year=[2]) 

# 4. Initialize an instance of SimpleRecursiveLattice for the option valuation.
#    - root: The initial value of the underlying asset/project for the lattice.
#    - Maximisation: The Maximisation_class instance created above.
#    - Rf: Risk-free rate.
#    - SubSteps: Number of sub-steps for the binomial lattice.
simple_tree=SimpleRecursiveLattice(root=445.08,
                                   Maximisation=Prototype_simple_option,
                                   Rf=0.034314,
                                   SubSteps=4)

# 5. Calculate and print the Expanded Net Present Value (ENPV) of the simple option contract.
#    - current_root: The starting value for the ENPV calculation, typically the root of the lattice.
#    - Current_L: The current level (index) in the STD array to use for volatility.
#                 Starting at 0 for the first level.
print("The expected option value for the simple option contract is "+str(simple_tree.Calculate_ENPV(current_root=simple_tree.root,Current_L=0).__round__(1))+" Million Dollar")