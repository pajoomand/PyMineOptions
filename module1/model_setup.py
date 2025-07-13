import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.io import loadmat
# --- Global Project Parameters ---
project_years = 34 # Defines the total duration of the project in years
first_year = 5 # Specifies a reference year, potentially for calculation start points like depreciation
x = loadmat('data/cashflow_fixed_parameters.mat')
class Prototype_Cash_flow():
    """
    Models the cash flow for a mining project, computing yearly Free Cash Flow (FCF)
    and the Present Value of Operating Cash Flows (PV_V0) and Total Project Value (Total_PVCF).
    It incorporates both fixed and simulated variable inputs.
    """
    def __init__(self, CF_inputs, project_years, fund_info, simulation_state):
        """
        Initialises the cash flow calculation engine.

        Args:
            CF_inputs (cash_flow_inputs): An object holding all necessary cash flow input variables.
            project_years (int): The total duration of the project in years.
            fund_info (Fund_info): An object containing funding details and WACC.
            simulation_state (int): A flag (0 or 1) indicating whether the calculation
                                    is for a base case (0) or a simulation run (1),
                                    affecting how discount factors are applied.
        """
        self.CF_inputs = CF_inputs
        self.project_years = project_years
        self.fund_info = fund_info
        self.simulation_state = simulation_state 
        
    def Calculate(self):
        """
        Performs the detailed calculation of yearly cash flows, including revenues,
        operating costs, capital expenditures, depreciation, tax, and working capital.
        It returns the Present Value of Operating Cash Flows (PV_V0) and
        the Total Present Value of Free Cash Flow (Total_PVCF).
        """
        # --- Initialising mine production assumptions across 3 stages/rows and project years ---
        mine_prod = np.ones((3, project_years)) 
        mine_prod[1][:] *= .8 # Adjusts production for the second stage/row
        mine_prod[2][:] *= .5 # Adjusts production for the third stage/row

        # --- Loading additional fixed cash flow inputs from the pre-loaded MATLAB file 'x' ---
        Unit_cost_fixed = x["Unit_cost_fixed_2_3"] # Fixed unit cost
        Corporate_fixed = x["Corporate_fixed"] # Fixed corporate overheads
        CHPP_fixed = x["CHPP_fixed_2_3"] # Fixed costs related to the Coal Handling and Preparation Plant
        #=========== End loading fixed cashflow inputs ============

        # --- Calculating Spot Revenue Components ---
        # ROM (Run-of-Mine) spot component, incorporating fixed value, spot percentage, and simulated variation
        ROM_spot = self.CF_inputs.ROM.fixed * self.CF_inputs.ROM.spot_percentage * (1 + self.CF_inputs.ROM.simulated) * mine_prod
        
        # Re-initialise mine_prod. This might overwrite previous adjustments if mine_prod
        # is intended to be a dynamic variable across different calculations.
        # It's important to verify if this reset is intentional for both spot and future calculations.
        mine_prod = np.ones((3, project_years))
        mine_prod[1, :] = mine_prod[1, :] * .8
        mine_prod[2, :] = mine_prod[2, :] * .5
        
        # Yield spot component, incorporating fixed value and simulated variation
        Yield_spot = self.CF_inputs.Yield.fixed * (1 + np.array(self.CF_inputs.Yield.simulated))
        Clean_spot = Yield_spot * ROM_spot # Calculate 'Clean' product from Yield and ROM
        Clean_total_spot = np.sum(Clean_spot, axis=0) # Sum 'Clean' product across stages for total spot
        # Calculate spot revenues based on price, exchange rate, and total clean product
        Revenues_spot = np.array(self.CF_inputs.Price.fixed / self.CF_inputs.Exchange_Rate.fixed) * np.array(Clean_total_spot)

        # --- Calculating Future Revenue Components ---
        # ROM future component, based on fixed value and remaining percentage after spot
        ROM_future = self.CF_inputs.ROM.fixed * (1 - self.CF_inputs.ROM.spot_percentage) * mine_prod
        Yield_future = self.CF_inputs.Yield.fixed # Yield assumed fixed for future component
        Price_future = np.ones(project_years) * 50 # Assumed fixed future price per unit
        Exchange_Rate_future = np.ones(project_years) * 0.7 # Assumed fixed future exchange rate
        Clean_future = Yield_future * ROM_future # Calculate 'Clean' product for future sales
        Clean_total_future = np.sum(Clean_future, axis=0) # Sum 'Clean' product across stages for total future
        # Calculate future revenues based on assumed future price, exchange rate, and total clean product
        Revenues_future = Price_future / Exchange_Rate_future * Clean_total_future
        
        # --- Total Revenues and Quantities (Spot + Future) ---
        Revenues = Revenues_spot + Revenues_future
        ROM = ROM_spot + ROM_future
        Clean_total = Clean_total_future + Clean_total_spot
        
        # --- Operating Costs (OPEX) Calculation ---
        OPEX_total = np.sum(Unit_cost_fixed * ROM, axis=0) # Total OPEX based on fixed unit cost and ROM
        CHPP_variable = CHPP_fixed * Clean_total # Variable CHPP costs based on total clean product
        # Total operational costs, incorporating OPEX simulation
        Operation = (OPEX_total + CHPP_variable) * (1 + self.CF_inputs.OPEX.simulated)
        # Variable transport costs, incorporating transport simulation
        Transport_variable = Clean_total * (self.CF_inputs.Transport.fixed * (1 + self.CF_inputs.Transport.simulated))
        # Variable port costs, incorporating port simulation
        Port_variable = Clean_total * (self.CF_inputs.Port.fixed * (1 + self.CF_inputs.Port.simulated))
        Royalty_variable = (Revenues - Port_variable) * self.CF_inputs.Royalty_fixed # Royalties based on adjusted revenues
        Marketing_variable = Revenues * self.CF_inputs.Marketing_fixed # Marketing costs as a percentage of revenues
        
        # Total Free On Board (FOB) cost, summing all operating and fixed costs, plus debt interest
        Total_FOB_cost = Operation[0, :] + Transport_variable + Port_variable + Royalty_variable + Marketing_variable + Corporate_fixed[0, :] + self.fund_info.debt_interest_amount
        
        # --- Capital Expenditure (CAPEX) Calculation ---
        # Sum of fixed CAPEX components from the MATLAB file
        total_capex_fixed = np.sum(self.CF_inputs.Capex.fixed, axis=0) + x["SpineAndCHPP_2_3"] + x["Exploration_fixed"]
        # Variable CAPEX, incorporating CAPEX simulation
        capex_variable = total_capex_fixed[0, :] * (1 + self.CF_inputs.Capex.simulated)
        
        # --- Calculating Depreciation and Tax Shield ---
        Depreciation = np.zeros((self.project_years)) # Array to store annual depreciation
        Taxsheild = np.zeros((self.project_years)) # Array to store annual tax shield
        
        # Depreciation calculation: 10% of cumulative CAPEX for first 10 years, then 10% of prior 10 years' CAPEX
        for year in range(4, 11): # From year 4 to year 10 (inclusive)
            Depreciation[year] = .1 * np.sum(capex_variable[0:year]) # 10% of cumulative capex up to current year
        for year in range(11, self.project_years): # From year 11 onwards
            Depreciation[year] = .1 * np.sum(capex_variable[year-10:year]) # 10% of capex from the preceding 10 years

        Taxable_income = Revenues - Total_FOB_cost # Income before tax and depreciation, but after FOB costs
        
        CForw_Loss = np.zeros((self.project_years)) # Array to track carry-forward losses
        Close_Balance = np.zeros((self.project_years)) # Balance used for tax calculation, considering losses

        # Loop to calculate carry-forward losses
        for year in range(1, self.project_years):
            # Calculate the closing balance for the year for tax purposes
            Close_Balance[year] = Taxable_income[year] - Depreciation[year] + CForw_Loss[year-1]
            if Close_Balance[year] < 0:
                CForw_Loss[year] = Close_Balance[year] # If balance is negative, it becomes a carry-forward loss

        # Loop to calculate the Tax Shield
        for year in range (1, self.project_years):
            if(Taxable_income[year] > 0): # Only if there's positive taxable income
                if ((Taxable_income[year] - Depreciation[year] + CForw_Loss[year-1]) > 0):
                    # If the adjusted taxable income is positive, tax shield is from depreciation and prior losses
                    Taxsheild[year] = Depreciation[year] - CForw_Loss[year-1]
                else:
                    # Otherwise, tax shield is limited to the current taxable income
                    Taxsheild[year] = Taxable_income[year]
        Taxsheild = Taxsheild * self.CF_inputs.Tax_Rate # Apply the corporate tax rate to the calculated tax shield

        # --- Calculating G (Adjustments, primarily Working Capital changes) ---
        # Calculation of working capital based on revenues and operations
        working_capital_calc = (Revenues / 365 * self.CF_inputs.Working_capital_days) - (Operation / 365 * self.CF_inputs.Working_capital_days)
        working_capital_calc = working_capital_calc[0, :] # Assumes working capital calculation is based on the first row if multi-dimensional
        
        Working_capital = np.zeros((self.project_years)) # Array to store annual working capital
        for year in range(1, self.project_years):
            if working_capital_calc[year] >= 0:
                Working_capital[year] = working_capital_calc[year] # Only positive working capital is considered

        Previous_year_work_capital = np.zeros((self.project_years)) # Array to store previous year's working capital
        for year in range (1, self.project_years):
            if Working_capital[year] == 0:
                Previous_year_work_capital[year] = 0
            else:
                Previous_year_work_capital[year] = Working_capital[year-1]
        
        # G represents total adjustments, including changes in working capital and CAPEX outflows
        G = Previous_year_work_capital - capex_variable - Working_capital

        # --- Calculating CF (Free Cash Flow) and V0 (Present Value of Operating Cash Flows) ---
        # Income After Tax, ensuring it's not negative
        Income_after_tax = Taxable_income - self.CF_inputs.Tax_Rate * Taxable_income
        for v in range (len(Income_after_tax)):
            if Income_after_tax[v] < 0:
                Income_after_tax[v] = 0
        
        # H is the Operating Cash Flow before considering adjustments (G)
        H = Income_after_tax + np.minimum(Taxable_income, np.zeros((len(Taxable_income))))
        
        # Free Cash Flow (CF) is Operating Cash Flow (H) plus Tax Shield and Adjustments (G)
        CF = H + Taxsheild + G
        
        # V0 is defined as the Present Value (PV) of the project’s net after-tax operating cash-flows
        # without performing the 'G' adjustments (e.g., working capital changes, capex outflows).
        V0 = H + Taxsheild 
        
        # --- Discounting Cash Flows to Present Value ---
        Discount_factor = np.zeros((self.project_years)) # Array to store discount factors for each year
        for year in range (self.project_years):
            Discount_factor[year] = 1 / (1 + self.fund_info.WACC)**(year + 1) # Discount factor using WACC
        
        # Adjust discount factors if in a simulation state (e.g., for Real Options, shifting time zero)
        if (self.simulation_state):
            shift(Discount_factor, 1) # Shifts array elements by 1 position (e.g., to align with option exercise)
            Discount_factor[0] = 1 # Sets the first element to 1, implying PV at time 0
        
        # Calculate Present Values by multiplying cash flows with discount factors
        PV_V0 = V0 * Discount_factor
        NPV = CF * Discount_factor # Calculate Net Present Value for Free Cash Flow
        
        PV_V0 = np.sum(PV_V0, axis=0) # Sum PV_V0 over all years to get total present value
        Total_PVCF = np.sum(NPV, axis=0) # Sum NPV (Total PV of Free Cash Flow) over all years

        return (PV_V0, Total_PVCF) # Return both PV_V0 and Total_PVCF

#==========================================================================================================================================

    def __init__(self,CF_inputs,project_years,fund_info,simulation_state):
        self.CF_inputs=CF_inputs
        self.project_years=project_years
        self.fund_info=fund_info
        self.simulation_state=simulation_state   
    def Calculate(self):
        # calculating  spot revenue
        mine_prod=np.ones((3,project_years))   
        mine_prod[1][:]*=.8
        mine_prod[2][:]*=.5

        # loading some other cashflow inputs from matlabfile
        Unit_cost_fixed=x["Unit_cost_fixed_2_3"]
        Corporate_fixed=x["Corporate_fixed"]
        CHPP_fixed=x["CHPP_fixed_2_3"]
        #===========end loading fixed self.CF_inputs============
        ROM_spot=self.CF_inputs.ROM.fixed*self.CF_inputs.ROM.spot_percentage*(1+self.CF_inputs.ROM.simulated)*mine_prod
        mine_prod=np.ones((3,project_years))
        mine_prod[1,:]=mine_prod[1,:]*.8
        mine_prod[2,:]=mine_prod[2,:]*.5
        Yield_spot=self.CF_inputs.Yield.fixed*(1+np.array(self.CF_inputs.Yield.simulated))
        Clean_spot=Yield_spot*ROM_spot
        Clean_total_spot=np.sum(Clean_spot,axis=0)
        Revenues_spot=np.array(self.CF_inputs.Price.fixed/self.CF_inputs.Exchange_Rate.fixed)* np.array(Clean_total_spot)
        # ===========calculating future revenue==========
        ROM_future=self.CF_inputs.ROM.fixed*(1-self.CF_inputs.ROM.spot_percentage)*mine_prod
        Yield_future=self.CF_inputs.Yield.fixed
        Price_future=np.ones(project_years)*50
        Exchange_Rate_future=np.ones(project_years)*0.7
        Clean_future=Yield_future*ROM_future
        Clean_total_future=np.sum(Clean_future,axis=0)
        Revenues_future=Price_future/Exchange_Rate_future* Clean_total_future
        Revenues=Revenues_spot+Revenues_future
        ROM=ROM_spot+ROM_future
        Clean_total=Clean_total_future+Clean_total_spot
        OPEX_total=np.sum(Unit_cost_fixed*ROM,axis=0)
        CHPP_variable=CHPP_fixed*Clean_total
        Operation=(OPEX_total+CHPP_variable)*(1+self.CF_inputs.OPEX.simulated)
        Transport_variable=Clean_total*(self.CF_inputs.Transport.fixed*(1+self.CF_inputs.Transport.simulated))
        Port_variable=Clean_total*(self.CF_inputs.Port.fixed*(1+self.CF_inputs.Port.simulated))
        Royalty_variable=(Revenues-Port_variable)*self.CF_inputs.Royalty_fixed
        Marketing_variable=Revenues*self.CF_inputs.Marketing_fixed
        Total_FOB_cost=Operation[0,:]+Transport_variable+Port_variable+Royalty_variable+Marketing_variable+Corporate_fixed[0,:]+self.fund_info.debt_interest_amount
        total_capex_fixed=np.sum(self.CF_inputs.Capex.fixed,axis=0)+x["SpineAndCHPP_2_3"]+x["Exploration_fixed"]
        capex_variable=total_capex_fixed[0,:]*(1+self.CF_inputs.Capex.simulated)
        Depreciation=np.zeros((self.project_years))
        Taxsheild=np.zeros((self.project_years))
        #-------------------------calculating taxsheild---------------------------

        for year in range (4,11):
            Depreciation[year]=.1*np.sum(capex_variable[0:year])
        for year in range (11,self.project_years):
            Depreciation[year]=.1*np.sum(capex_variable[year-10:year])
        Taxable_income=Revenues-Total_FOB_cost
        CForw_Loss=np.zeros((self.project_years))
        Close_Balance=np.zeros((self.project_years))
        for year in range(1,self.project_years):
            Close_Balance[year]=Taxable_income[year]-Depreciation[year]+CForw_Loss[year-1]
            if Close_Balance[year]<0:
                CForw_Loss[year]=Close_Balance[year]
        for year in range (1,self.project_years):
            if(Taxable_income[year]>0):
                if ((Taxable_income[year]-Depreciation[year]+CForw_Loss[year-1])>0):
                    Taxsheild[year]=Depreciation[year]-CForw_Loss[year-1]
                else:
                    Taxsheild[year]=Taxable_income[year]
        Taxsheild=Taxsheild*self.CF_inputs.Tax_Rate
        #-----------------------calculating G as adjustments--------------------
        working_capital_calc=(Revenues/365*self.CF_inputs.Working_capital_days)-(Operation/365*self.CF_inputs.Working_capital_days)
        working_capital_calc=working_capital_calc[0,:]
        Working_capital=np.zeros((self.project_years))
        for year in range(1,self.project_years):
            if working_capital_calc[year]>=0:
                 Working_capital[year]=working_capital_calc[year]
        Previous_year_work_capital=np.zeros((self.project_years))
        for year in range (1,self.project_years):
            if Working_capital[year]==0:
                Previous_year_work_capital[year]=0
            else:
                Previous_year_work_capital[year]=Working_capital[year-1]
        G=Previous_year_work_capital-capex_variable-Working_capital

        # -------------------------calculating CF and V0--------------------        
        Income_after_tax=Taxable_income-self.CF_inputs.Tax_Rate*Taxable_income
        for v in range (len(Income_after_tax)):
            if Income_after_tax[v]<0:
                Income_after_tax[v]=0
        H=Income_after_tax+np.minimum(Taxable_income,np.zeros((len(Taxable_income))))
        CF=H+Taxsheild+G
        V0=H+Taxsheild      
        #is defined as the PV of the project’s net after tax operating cash-flows without performing adjustments. 
        #CF(maturity)=CF(maturity)-debt; #we now assume that we are paying debt for
        #life time that is 30 years in EQPV tree
        Discount_factor=np.zeros((self.project_years))
        for year in range (self.project_years):
            Discount_factor[year]=1/(1+self.fund_info.WACC)**(year+1)
        if (self.simulation_state):
            shift(Discount_factor,1)
            Discount_factor[0]=1
        PV_V0=V0*Discount_factor
        NPV=CF*Discount_factor
        PV_V0=np.sum(PV_V0,axis=0)
        Total_PVCF=np.sum(NPV,axis=0)
        return (PV_V0,Total_PVCF)