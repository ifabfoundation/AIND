from utils import *

class MAE_CI():
    def __init__(self):
        self.mae = dict()

    def calculate(self, d_real, d_predicted):
        for var in d_real.columns:
            residuals = np.asarray(d_real[var]) - np.asarray(d_predicted[var])
            error = np.mean(np.abs(residuals))

            # To obtain theconfidence intervall we need:
            # - the std of the residuals
            res_std = np.std(residuals)
            # - length of the predicted data
            res_len = len(residuals)
            # - z for the 95% of coinfidence
            z = 1.96

            # Confidence Interval
            error_interval = z*res_std/np.sqrt(res_len)
            
            # Store result
            self.mae[var] = dict()
            # Store MAE
            self.mae[var]['mae'] = error
            # Store error intervak
            self.mae[var]['ci'] = error_interval
        
        return self.mae
