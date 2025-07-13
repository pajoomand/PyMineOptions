class Fund_info():
    """
    Manages financial information related to project funding,
    including equity, debt, interest rates, and the Weighted Average Cost of Capital (WACC).
    """
    def __init__(self, fund_percentage, fund_amount, interest_rate, nominal_discount_rate, maturity):
        """
        Initialises funding details.

        Args:
            fund_percentage (float): The proportion of total funding that is equity (e.g., 0.6 for 60% equity).
            fund_amount (float): The total amount of project funding.
            interest_rate (float): The annual interest rate on debt.
            nominal_discount_rate (float): The nominal discount rate for equity (cost of equity).
            maturity (int): The maturity period of the debt in years.
        """
        self.fund_percentage = fund_percentage
        self.fund_amount = fund_amount
        self.interest_rate = interest_rate
        self.nominal_discount_rate = nominal_discount_rate
        self.maturity = maturity
    
    @property
    def equity(self):
        """Calculates the total equity amount based on fund percentage and total fund amount."""
        return self.fund_percentage * self.fund_amount
    
    @property
    def debt(self):
        """Calculates the total debt amount as the remainder of total funds after equity."""
        return self.fund_amount - self.equity
    
    @property
    def debt_interest_amount(self):
        """Calculates the annual interest payment on the debt."""
        return self.debt * self.interest_rate
    
    @property
    def WACC(self):
        """
        Calculates the Weighted Average Cost of Capital (WACC).
        This is a common discount rate used in financial modelling to value projects.
        It assumes the nominal_discount_rate is the cost of equity and debt_interest_amount
        is the annual cost of debt before tax effects (as tax shield is handled elsewhere).
        """
        return (self.equity * self.nominal_discount_rate + self.debt_interest_amount) / self.fund_amount

#==========================================================================================================================================
   