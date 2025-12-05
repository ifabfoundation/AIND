from utils import *

class MAE_CI():
    def __init__(self):
        self.mae = dict()

    def calculate(self, d_real, d_predicted):
        """
        Calculate the Mean Absolute Error and Confidence Interval.
        Handle the NaN values, calculate the MAE and CI only on valid values.

        Parameters
        ----------
        d_real : pandas.DataFrame
            Real data
        d_predicted : pandas.DataFrame
            Predicted data

        Returns
        -------
        mae_dict : dict
            Dictionary with MAE and CI for each variable
        """
        for var in d_real.columns:
            # Take the real and predicted values
            real_values = np.asarray(d_real[var])
            predicted_values = np.asarray(d_predicted[var])
            
            # Create a mask to identify non-NaN values
            mask = ~np.isnan(real_values)
            
            # Filter values using the mask
            real_filtered = real_values[mask]
            pred_filtered = predicted_values[mask]
            
            # Calculate residuals only on valid values
            residuals = real_filtered - pred_filtered
            
            # If there are no valid values, set error and ci to NaN
            if len(residuals) == 0:
                error = np.nan
                error_interval = np.nan
            else:
                # Calculate MAE
                error = np.mean(np.abs(residuals))
                
                # Calculate the confidence interval
                # - std of residuals
                res_std = np.std(residuals)
                # - length of predicted data (filtered)
                res_len = len(residuals)
                # - z for 95% confidence
                z = 1.96
                
                # Calculate the confidence interval
                error_interval = z*res_std/np.sqrt(res_len)
            
            # Store result
            self.mae[var] = dict()
            # Store MAE
            self.mae[var]['mae'] = error
            # Store error interval
            self.mae[var]['ci'] = error_interval
            # Store the number of valid samples used for the calculation
            self.mae[var]['valid_samples'] = len(residuals)
        
        return self.mae
