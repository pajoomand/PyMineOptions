project_years=34
first_year=5
import random
import numpy as np
class Input__risk_variable():
    """
    Represents a single input risk variable that influences the project's valuation.
    Each variable can be defined by its name, distribution parameters, a fixed value,
    and can hold a simulated value or an output sigma from analysis.
    """
    def __init__(self, name, distribution, fixed, simulated=0, output_sigma=0):
        """
        Initialises an input risk variable.

        Args:
            name (str): The unique name of the risk variable (e.g., "ROM", "Yield", "Price").
            distribution (Distribution object): An instance of the Distribution class,
                                               containing parameters for the variable's
                                               probability distribution (mean, tau, type, variation).
            fixed (float/np.array): The base or fixed value of the variable.
            simulated (float/np.array, optional): The value generated during simulation. Defaults to 0.
            output_sigma (float, optional): The calculated output sigma (volatility) for this variable. Defaults to 0.
        """
        self.name = name
        self.distribution = distribution
        self.output_sigma = output_sigma
        self.fixed = fixed
        self.simulated = simulated
    
    def generate_randome_number(self):
        """
        Generates a random number for the variable based on its defined probability distribution.
        This method updates the 'simulated' or 'fixed' attribute of the instance.
        Assumes statistical independence between variables for simulation purposes.
        """
        if (self.name == "ROM") or (self.name == "Yield"):
            # For "ROM" (Run-of-Mine) and "Yield", a triangular distribution is used.
            # 'new_tau' adjusts the scale parameter based on 'tau' and 'tau_variation'.
            new_tau = (self.distribution.tau / ((6)**0.5)) * (1 + self.distribution.tau_vairation)
            # 'boundary' is calculated but not directly used in the np.random.triangular call.
            boundary = new_tau * (6**0.5) 
            lower = -self.distribution.tau # Lower bound of the triangular distribution
            peak = 0 # Peak (mode) of the triangular distribution, indicating the most likely value
            upper = self.distribution.tau # Upper bound of the triangular distribution
            # Generates a single random number from a triangular distribution and
            # assigns it to a 3xproject_years NumPy array for simulation across stages/years.
            self.simulated = np.ones((3, project_years)) * float(np.random.triangular(lower, peak, upper)).__round__(5)
        elif (self.name == "Exchange_Rate" or self.name == "Price"):
            # For "Exchange_Rate" and "Price", a normal distribution is used.
            # The 'fixed' attribute is updated with a random number from this distribution.
            # This implies 'fixed' might represent a base value that gets perturbed during simulation.
            self.fixed = np.random.normal(self.distribution.mean, self.distribution.tau * (1 + self.distribution.tau_vairation))

class input_risk_variable_with_spot_percentage(Input__risk_variable):
    """
    Extends the Input__risk_variable class to include a 'spot_percentage'.
    This is relevant for variables that have both an immediate (spot) component
    and a future component in the cash flow model.
    """
    def __init__(self, name, distribution, fixed, simulated, output_sigma, spot_percentage):
        """
        Initialises an input risk variable with an additional spot percentage.

        Args:
            spot_percentage (float): The percentage of the variable's value
                                     that is considered to be 'spot' (immediate).
        """
        super().__init__(name, distribution, fixed, simulated, output_sigma)
        self.spot_percentage = spot_percentage
