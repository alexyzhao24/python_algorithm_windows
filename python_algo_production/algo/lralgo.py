import numpy as np

class LinearRegressionHistogram:
    """
    A production-grade utility to compute linear regression fits 
    and evaluate value distributions via histogram binning arrays.
    """
    @staticmethod
    def compute_lr_hist(data_x, data_y, bins=10):
        """
        Calculates a simple linear regression line (y = mx + c) for the inputs,
        then bins the residuals (error metrics) into a distribution histogram.
        """
        x = np.array(data_x, dtype=np.float64)
        y = np.array(data_y, dtype=np.float64)
        
        if len(x) < 2 or len(x) != len(y):
            raise ValueError("Input arrays must be of equal length and contain at least 2 points.")
            
        # Calculate Slope (m) and Intercept (c) using least squares
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate residuals (predicted y minus actual y)
        predictions = m * x + c
        residuals = y - predictions
        
        # Compute histogram distribution of the residuals
        counts, bin_edges = np.histogram(residuals, bins=bins)
        
        return {
            "slope": float(m),
            "intercept": float(c),
            "histogram_counts": counts.tolist(),
            "bin_edges": bin_edges.tolist()
        }