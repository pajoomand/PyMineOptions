class Distribution():
    """
    A simple class to define the parameters for a probability distribution
    associated with an input risk variable.
    """
    def __init__(self, mean, tau, dis_type, tau_vairation=0):
        """
        Initialises the distribution parameters.

        Args:
            mean (float): The mean of the distribution.
            tau (float): A measure of spread or volatility. For normal, it's std dev;
                         for triangular, it relates to the bounds.
            dis_type (str): The type of distribution (e.g., "normal", "triangular").
            tau_vairation (float, optional): A percentage variation applied to 'tau',
                                             allowing for uncertainty in the volatility itself. Defaults to 0.
        """
        self.mean = mean
        self.tau = tau
        self.dis_type = dis_type
        self.tau_vairation = tau_vairation
#==========================================================================================================================================
