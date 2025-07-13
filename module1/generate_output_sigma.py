import random
import numpy as np
import math 


class generate_output_sigma():
    """
    Performs Monte Carlo simulations to quantify the effect of individual
    random input risk variables on the overall project value volatility (output sigma).
    Each output sigma measures the standard deviation of changes in the natural
    logarithm of the project value (log-returns).
    """
    def __init__(self, cashflow, simulated_variable, iterations):
        """
        Initialises the Monte Carlo simulation process for output sigma generation.

        Args:
            cashflow (Prototype_Cash_flow): An instance of the cash flow model to be simulated.
            simulated_variable (list of str): A list containing the name(s) of the
                                              input risk variable(s) to be simulated
                                              in this specific run (e.g., ["ROM"]).
            iterations (int): The number of Monte Carlo simulation runs to perform.
        """
        self.cashflow = cashflow
        self.simulated_variable = simulated_variable
        self.iterations = iterations
        
    def MC_simulation(self):
        """
        Executes the Monte Carlo simulations.
        It first calculates a base case NPV, then iteratively simulates the
        specified risk variable, recalculates NPV, and finally computes the
        standard deviation of the log-returns of the project value.
        """
        # --- Reset all simulated variables to their fixed/base case values ---
        # This ensures that only the 'simulated_variable' chosen for this run is random.
        self.cashflow.CF_inputs.ROM.simulated = 0
        self.cashflow.CF_inputs.Yield.simulated = 0
        self.cashflow.CF_inputs.Exchange_Rate.simulated = 0
        self.cashflow.CF_inputs.Price.simulated = 0
        self.cashflow.simulation_state = 0 # Set simulation state to base case (not active simulation)
        
        # Calculate the base case NPV (Project Value) before any specific variable simulation
        # 'tmp' here captures PV_V0, 'NPV' captures Total_PVCF from the base calculation.
        [tmp, NPV] = self.cashflow.Calculate() 
        
        self.cashflow.simulation_state = 1 # Set simulation state to active simulation for subsequent calls
        Y = np.zeros(self.iterations) # Array to store the Total_PVCF results from each simulation run
        
        # --- Run Monte Carlo iterations ---
        for r in range (self.iterations):
            # Generate random numbers ONLY for the specified simulated variable(s)
            if "ROM" in self.simulated_variable:
                self.cashflow.CF_inputs.ROM.generate_randome_number()
            if "Yield" in self.simulated_variable:
                self.cashflow.CF_inputs.Yield.generate_randome_number()
            if "Exchange_Rate" in self.simulated_variable:
                self.cashflow.CF_inputs.Exchange_Rate.generate_randome_number()
            if "Price" in self.simulated_variable:
                self.cashflow.CF_inputs.Price.generate_randome_number()

            # Calculate the project value (Total_PVCF) for the current simulation run
            # The [1] index retrieves Total_PVCF from the tuple returned by Calculate().
            Y[r] = (self.cashflow.Calculate()[1])
            
        # Calculate the ratio of simulated NPVs to the base case NPV
        tmp = Y / NPV 
        c1 = (tmp) > 0 # Create a boolean mask to filter out non-positive values (required for log calculation)
        
        # Calculate the natural logarithm of the positive ratios (log-returns)
        PVOPCF = np.log(tmp[c1]) 
        # Return the standard deviation of these log-returns, which is the output sigma for this variable
        return(np.std(PVOPCF))
