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


print("The output sigma for the selected input risk variable is "+str(Yield_output_sigma))