import numpy as np
import math


#==========================================================================================================================================

class Maximisation_class():
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

    # The commented-out properties are left as they were, as per your previous note on performance.
    # # We don't use property feature of the class as it slows down the program
    # # @property
    # # def investments(self):
    # # ...
    # # @property
    # # def T(self):
    # # ...