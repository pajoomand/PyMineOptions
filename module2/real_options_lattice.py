import numpy as np
import math

class RecursiveLattice:
    """
    Base class for building and valuing options using a recursive binomial lattice (decision tree).
    It defines common properties and static methods for lattice operations.
    """
    def __init__(self, root, Maximisation, Rf, SubSteps):
        """
        Initializes the recursive lattice.

        Args:
            root (float): The initial value of the underlying asset (e.g., project value) at the root node.
            Maximisation (Maximisation_class object): An instance of Maximisation_class containing
                                                      parameters related to option exercise,
                                                      volatilities (STD), investments, etc.
            Rf (float): The risk-free rate.
            SubSteps (int): The number of time steps (sub-divisions) within each main period (level).
                            This increases the granularity of the lattice.
        """
        self.root = root
        self.Maximisation = Maximisation # Object holding option-specific parameters
        self.Rf = Rf # Risk-free rate
        self.SubSteps = SubSteps # Number of small time steps per main period/level

    @staticmethod
    def LatticeNodes(L):
        """
        Calculates the total number of nodes up to a given level L in a standard
        binomial lattice. For a level L, there are L+1 nodes.
        The formula (L+1)*(L+2)/2 gives the sum of nodes from level 0 to L.
        This is used for indexing a 1D array representation of the lattice.

        Args:
            L (int): The current level of the lattice.

        Returns:
            int: The total count of nodes from level 0 up to level L,
                 which also corresponds to the index of the last node at level L
                 in a specific 1D array mapping.
        """
        return int((L + 1) * (L + 2) / 2)

    @staticmethod
    def discount_factor(rf, delta_t):
        """
        Calculates the continuous compounding discount factor.

        Args:
            rf (float): The risk-free rate.
            delta_t (float): The time increment.

        Returns:
            float: The discount factor.
        """
        return (math.exp(-rf * delta_t))

class SimpleRecursiveLattice(RecursiveLattice):
    """
    Implements a single-asset, recursive binomial lattice for valuing a real option.
    It considers a sequence of volatilities (STD) for different "levels" or periods
    of the option's life, and performs a single maximisation at the terminal level.
    """
    def __init__(self, root, Maximisation, Rf, SubSteps):
        """
        Initializes the simple recursive lattice, inheriting from RecursiveLattice.
        """
        super().__init__(root, Maximisation, Rf, SubSteps)

    def Calculate_ENPV(self, current_root, Current_L):
        """
        Calculates the Expanded Net Present Value (ENPV) using a recursive multi-level
        binomial lattice approach. Each level can have a different volatility.
        This function handles the forward construction of a lattice segment and
        the backward rollback for valuation.

        Args:
            current_root (float): The value of the current node from which this
                                  recursive call is building a sub-lattice.
            Current_L (int): The current level index in the sequence of volatilities (STD).
                             This determines which sigma to use for the current sub-lattice.

        Returns:
            float: The calculated ENPV at the 'current_root' of this sub-lattice.
        """
        # Retrieve the specific volatility for the current level
        # Assumes self.Maximisation.STD is an array/list of volatilities for each level.
        sigma = self.Maximisation.STD[Current_L]

        # Calculate time step (dt) for the sub-steps within the current level.
        # T[0] is assumed to be the total time period for all levels.
        # dt is T[0] divided by (number of volatility levels * number of sub-steps per level)
        dt = self.Maximisation.T[0] / (len(self.Maximisation.STD) * self.SubSteps)

        # Calculate Up (U) and Down (D) factors, and risk-neutral probability (p)
        U = math.exp(sigma * dt**0.5)
        D = 1 / U
        p = (math.exp(self.Rf * dt) - D) / (U - D)

        # Initialize a 1D NumPy array to store the nodes of the current sub-lattice.
        # The size is based on the total number of nodes up to self.SubSteps (last sub-level).
        ForwardLattice = np.zeros(self.LatticeNodes(self.SubSteps) + 1)
        Pn = current_root # Pn represents the 'parent' node for building the current level

        # Set the root of the current sub-lattice
        ForwardLattice[1] = Pn # Node 1 is typically the root in this 1D mapping

        Total_levels = len(self.Maximisation.STD) - 1 # Total number of volatility levels (0-indexed)

        # --- Forward Construction of the Sub-Lattice ---
        # Iterate through each sub-step (level) in the current sub-lattice
        for i in range(1, self.SubSteps + 1):
            # Calculate the "Down" node for the current level 'i'.
            # LatticeNodes(i) gives the index of the last node in level 'i'.
            # In this 1D mapping, Pn is the 'parent' node that leads to the down path.
            ForwardLattice[self.LatticeNodes(i)] = Pn * D
            # Update Pn to the newly calculated down node value. This structure means Pn
            # is always the 'latest' down node value from the *previous* main level.
            Pn = ForwardLattice[self.LatticeNodes(i)]

            # Calculate the "Up" nodes for the current level 'i'.
            # This loop iterates through the nodes of level 'i' from left to right (excluding the last 'down' node).
            # The indexing 'j-i' in ForwardLattice[j-i] is specific to this 1D lattice mapping.
            # It accesses the 'parent' node from the previous level (i-1) that leads to an 'up' path.
            # This mapping is crucial and should be thoroughly verified for robustness.
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i)):
                ForwardLattice[j] = ForwardLattice[j - i] * U

        # --- Recursive Calls or Terminal Node Maximisation ---
        # If there are more volatility levels (i.e., not at the overall option expiry level)
        if Current_L < Total_levels:
            # For each terminal node of the current sub-lattice, recursively call Calculate_ENPV
            # to build the next sub-lattice based on the next volatility level (Current_L + 1).
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i) + 1):
                ForwardLattice[j] = self.Calculate_ENPV(ForwardLattice[j], Current_L + 1)
        else:
            # This is the overall terminal decision level (option expiry date).
            # Perform maximisation for the final nodes.
            # The loop iterates through the nodes at the last sub-level (self.SubSteps)
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i) + 1):
                # Calculate the value after exercising the option, subtracting discounted investment cost.
                # Assumes investments[0] is the relevant nominal cost at T[0] for this single option.
                exercise_value = ForwardLattice[j] - (self.Maximisation.investments[0] / math.exp(self.Maximisation.NomDiscountFactor * self.Maximisation.T[0]))
                
                # The option value is max(exercise_value, 0) (cannot be negative)
                result = max(exercise_value, 0) # Use max() for clarity

                # IMPORTANT: Terminal nodes hold the intrinsic value at expiry.
                # Discounting happens during the rollback process, not here.
                ForwardLattice[j] = result

        # --- Rollback Towards the Root ---
        # Iterate backwards through the sub-steps to discount and combine node values.
        for i in range(self.SubSteps, 0, -1):
            # Iterate through the nodes of the previous level (i-1) to calculate their values
            # based on the values of their children in the current level (i).
            # The range (self.LatticeNodes(i-1)-i+1) to (self.LatticeNodes(i-1)+1)
            # covers the nodes at level (i-1) in this 1D mapping.
            for j in range((self.LatticeNodes(i - 1) - i + 1), self.LatticeNodes(i - 1) + 1):
                # Apply the risk-neutral valuation formula: (p * Up_Child + (1-p) * Down_Child)
                # The indices (j+i) and (j+i+1) refer to the up-child and down-child nodes respectively
                # in the current level 'i', from parent node 'j' in level 'i-1'.
                ForwardLattice[j] = ((ForwardLattice[j + i] * p) + (ForwardLattice[j + i + 1] * (1 - p)))

                # Discount the calculated node value back one time step (dt),
                # unless it's the very first root node (j=1) in the initial call (Current_L=0).
                # The root node itself is the final result and should not be discounted after its calculation.
                if not (j == 1 and Current_L == 0):
                    ForwardLattice[j] = ForwardLattice[j] * self.discount_factor(self.Rf, dt)

        # The value at index 1 is the value of the root node of the current sub-lattice.
        return ForwardLattice[1]

class CompoundRecursiveLattice(RecursiveLattice):
    """
    Implements a lattice for valuing compound real options, where the option to invest
    can itself lead to another option. It supports multiple stages of maximisation (investments).
    """
    def __init__(self, root, Maximisation, Rf, SubSteps):
        """
        Initializes the compound recursive lattice, inheriting from RecursiveLattice.
        """
        super().__init__(root, Maximisation, Rf, SubSteps)

    @staticmethod
    def NominalDiscountFactor(endpoint, maximisation, c):
        """
        Calculates a nominal discounted value related to equity and investments for a compound option.
        This function appears to represent a cash flow or value to equity holders
        at a specific node, considering a percentage of equity and an investment cost.

        Args:
            endpoint (float): The project value at a specific node (e.g., ForwardLattice[j]).
            maximisation (Maximisation_class object): Object containing option exercise parameters.
            c (int): Index for accessing specific investment/equity percentage in the Maximisation object.

        Returns:
            float: A calculated value representing the outcome of exercising a stage of the option.
        """
        # This calculates a value after considering equity percentage and investment cost
        return (endpoint * maximisation.Percentage_Equity[c] - (maximisation.investments[c]))

    def Calculate_ENPV(self, current_root, Current_L):
        """
        Calculates the Expanded Net Present Value (ENPV) for a compound option
        using a recursive multi-level binomial lattice. It handles multiple stages
        of investment/maximisation.

        Args:
            current_root (float): The value of the current node from which this
                                  recursive call is building a sub-lattice.
            Current_L (int): The current level index in the sequence of volatilities (STD)
                             and maximisation points.

        Returns:
            float: The calculated ENPV at the 'current_root' of this sub-lattice.
        """
        # Retrieve the specific volatility for the current level.
              # `sigma` will have twice the length of the original `self.Maximisation.STD`,
        # effectively repeating the volatility pattern for more levels of the compound option.
        sigma = self.Maximisation.STD + self.Maximisation.STD

        maxi_compound_level = len(self.Maximisation.STD) - 1 # Index of the last maximisation level for the *original* STD length

        # Calculate time step (dt) for the sub-steps within the current level.
        # dt is T[0] divided by (total number of *original* volatility levels * number of sub-steps per level).
        # Note: This `dt` calculation uses `len(self.Maximisation.STD)` (original length),
        # while `Total_levels` below uses `len(sigma)` (concatenated length).
        dt = self.Maximisation.T[0] / (len(self.Maximisation.STD) * self.SubSteps)

        # Calculate Up (U) and Down (D) factors, and risk-neutral probability (p)
        U = math.exp(sigma[Current_L] * dt**0.5) # sigma[Current_L] will use a value from the concatenated list
        D = 1 / U
        p = (math.exp(self.Rf * dt) - D) / (U - D)

        # Initialize 1D NumPy arrays for the current sub-lattice nodes and for
        # storing values specifically for compound maximisation.
        ForwardLattice = np.zeros(self.LatticeNodes(self.SubSteps) + 1)
        forward_maximisation = np.zeros(self.LatticeNodes(self.SubSteps) + 1)
        Pn = current_root

        # Set the root of the current sub-lattice
        ForwardLattice[1] = Pn

        # Total number of effective volatility levels for the entire tree, based on the concatenated sigma.
        Total_levels = len(sigma) - 1

        # --- Forward Construction of the Sub-Lattice ---
        for i in range(1, self.SubSteps + 1):
            ForwardLattice[self.LatticeNodes(i)] = Pn * D
            Pn = ForwardLattice[self.LatticeNodes(i)]
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i)):
                ForwardLattice[j] = ForwardLattice[j - i] * U

            # If this is the specific level where compound maximisation occurs,
            # store the current lattice values for later use in that maximisation step.
            if Current_L == maxi_compound_level:
                # This range should capture all nodes at the current sub-level 'i'.
                forward_maximisation[self.LatticeNodes(i-1)+1 : self.LatticeNodes(i)+1] = \
                    ForwardLattice[self.LatticeNodes(i-1)+1 : self.LatticeNodes(i)+1]
        
        # --- Recursive Calls or Terminal/Compound Maximisation ---
        # If there are more effective volatility levels to process recursively
        if Current_L < Total_levels:
            # Recursively call Calculate_ENPV for each terminal node of the current sub-lattice
            # to build the next sub-lattice.
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i) + 1):
                ForwardLattice[j] = self.Calculate_ENPV(ForwardLattice[j], Current_L + 1)

            # Check if this current level is the designated 'compound maximisation' level.
            # This check happens *after* the recursive calls return.
            if Current_L == maxi_compound_level:
                # Perform the compound option maximisation at this specific level.
                for v in range(self.LatticeNodes(i - 1) + 1, self.LatticeNodes(i) + 1):
                    # Calculate the value of exercising the compound option using NominalDiscountFactor
                    # Assumes '0' as the index 'c' for this specific maximisation stage investment.
                    compound_exercise_val = self.NominalDiscountFactor(forward_maximisation[v], self.Maximisation, 0)
                    
                    # Compare the value of exercising the compound option (compound_exercise_val)
                    # with the value of letting the underlying option expire (max(ForwardLattice[v] - investment, 0)).
                    # Uses investments[0] as the cost for the underlying option exercise comparison.
                    ForwardLattice[v] = max(compound_exercise_val, (max(ForwardLattice[v] - self.Maximisation.investments[0], 0)))
        else:
            # This is the overall terminal decision level (final option expiry).
            # Perform maximisation for the final nodes using the specified NominalDiscountFactor
            # with index '1' for investments/equity if applicable (for a second stage of investment).
            for j in range((self.LatticeNodes(i - 1) + 1), self.LatticeNodes(i) + 1):
                # Calculate the value upon final exercise.
                # Assumes '1' as the index 'c' for this final maximisation stage investment.
                temp = self.NominalDiscountFactor(ForwardLattice[j], self.Maximisation, 1)
                
                # Option value cannot be negative.
                result = max(temp, 0)

                # IMPORTANT: Terminal nodes hold the intrinsic value at expiry.
                # Discounting happens during the rollback process, not here.
                ForwardLattice[j] = result

        # --- Rollback Towards the Root ---
        # Iterate backwards through the sub-steps to discount and combine node values.
        for i in range(self.SubSteps, 0, -1):
            for j in range((self.LatticeNodes(i - 1) - i + 1), self.LatticeNodes(i - 1) + 1):
                # Apply the risk-neutral valuation formula
                ForwardLattice[j] = ((ForwardLattice[j + i] * p) + (ForwardLattice[j + i + 1] * (1 - p)))

                # Discount the calculated node value back one time step (dt),
                # unless it's the very first root node (j=1) in the initial call (Current_L=0).
                if not (j == 1 and Current_L == 0):
                    ForwardLattice[j] = ForwardLattice[j] * self.discount_factor(self.Rf, dt)

        # Return the value of the root node of the current sub-lattice.
        return ForwardLattice[1]

#==========================================================================================================================================


    """
    Manages parameters specific to the option's exercise and maximisation stages,
    including equity contribution, expenses, discount factors, volatilities,
    and the timing and cost of investments.
    """
    def __init__(self, equity, Percentage_Equity, expenses, NomDiscountFactor, STD, Year):
        """
        Initializes the maximisation parameters for option valuation.

        Args:
            equity (float): The total initial equity investment or project size for calculations.
            Percentage_Equity (list/array): A list/array of percentage equity contributions
                                            at different investment stages.
                                            Interpretation: This is assumed to be an *incremental*
                                            percentage for each stage if `(self.Percentage_Equity[i]-increased_equity)`
                                            is to calculate the additional equity for the current stage.
                                            Alternatively, if `Percentage_Equity[i]` is a cumulative percentage,
                                            the calculation `(self.Percentage_Equity[i]-increased_equity)` is correct,
                                            but then `increased_equity = self.Percentage_Equity[i]` should be used
                                            to update `increased_equity`.
                                            The current code's update `increased_equity=self.Percentage_Equity[i]+increased_equity`
                                            implies `Percentage_Equity[i]` is incremental, and `increased_equity`
                                            is the running sum of these increments. This creates a potential
                                            inconsistency with `(self.Percentage_Equity[i]-increased_equity)`.
                                            For current code, it means `invest[i]` uses `(current_incremental - previous_total_incremental)`.
                                            Please verify how `Percentage_Equity` is defined for `Maximisation_class`.
            expenses (list/array): A list/array of expenses associated with each investment stage.
            NomDiscountFactor (float): The nominal discount factor, possibly for nominal cash flows or costs.
            STD (list/array): A list/array of standard deviations (volatilities) for each level/stage
                              of the option or underlying project. These would typically come from Module 1.
            Year (list/array): A list/array indicating the years at which investment decisions/stages occur.
        """
        self.equity = equity
        self.expenses = expenses
        self.NomDiscountFactor = NomDiscountFactor
        self.STD = STD # Standard deviations/volatilities for different stages
        self.Percentage_Equity = Percentage_Equity # Percentage of equity at each stage
        self.Year = Year # Years when stages occur

        # --- Calculate T (Time to Option Expiry/Stage Duration) ---
        # 'T' (self.T) is calculated as the duration between consecutive investment years,
        # representing the time horizons for each stage of the option.
        t = [0] * len(self.Year)
        if len(self.Year) > 0: # Ensure Year is not empty
            t[0] = self.Year[0] # Duration for the first stage is the first year itself
            if (len(self.Year) > 1):
                for i in range(1, len(self.Year)):
                    t[i] = self.Year[i] - self.Year[i - 1] # Duration between current and previous year
        self.T = t

        # --- Calculate Investments for Each Option Stage (Present Value) ---
        # 'investments' (self.investments) stores the present value of the investment cost
        # associated with each stage of the option.
        invest = [0] * len(self.Year)
        increased_equity = 0 # Accumulator for total percentage equity committed so far

        for i in range(len(self.Year)):
            # Calculate the nominal investment amount at year[i]:
            # This calculation `(self.Percentage_Equity[i]-increased_equity)`
            # implies that `increased_equity` holds the *cumulative* percentage of equity from
            # *previous* stages, and `self.Percentage_Equity[i]` is the *cumulative percentage up to stage i*.
            # If `self.Percentage_Equity[i]` is meant to be the *incremental* percentage for stage i,
            # then this calculation of `additional_percentage` is incorrect.
            # Assuming `self.Percentage_Equity` is cumulative, and `increased_equity` is cumulative before current stage.
            additional_percentage_for_stage = self.Percentage_Equity[i] - increased_equity
            
            nominal_investment = additional_percentage_for_stage * self.equity + self.expenses[i]

            # Discount this nominal investment back to Year 0 (Present Value)
            # using the NominalDiscountFactor and the specific Year[i]
            invest[i] = nominal_investment / math.exp(self.NomDiscountFactor * self.Year[i])
            
            # Update the accumulated equity percentage for the *next* iteration.
            # This line: `increased_equity=self.Percentage_Equity[i]+increased_equity`
            # contradicts the assumption that `Percentage_Equity[i]` is cumulative (where it should be `increased_equity = self.Percentage_Equity[i]`).
            # If `Percentage_Equity[i]` is indeed incremental, then `additional_percentage_for_stage` should just be `self.Percentage_Equity[i]`,
            # and `increased_equity` should accumulate `increased_equity += self.Percentage_Equity[i]`.
            # Leaving as is based on original code, but noting this potential ambiguity.
            increased_equity = self.Percentage_Equity[i] + increased_equity 
        self.investments = invest

 