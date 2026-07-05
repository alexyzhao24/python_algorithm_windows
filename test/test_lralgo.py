import pytest
import numpy as np
from python_algo_production.algo.lralgo import LinearRegressionHistogram

def test_lr_hist_naive_basic():
    """Validates basic linear regression calculation behavior on fixed inputs."""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5] # General upward trend
    
    result = LinearRegressionHistogram.compute_lr_hist(x, y, bins=3)
    
    assert "slope" in result
    assert "intercept" in result
    assert len(result["histogram_counts"]) == 3
    assert sum(result["histogram_counts"]) == len(x)

def test_lr_hist_naive_random():
    """Validates that random noise distributions evaluate cleanly without nan limits."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50).tolist()
    y = (2 * np.array(x) + np.random.normal(0, 1, 50)).tolist()
    
    result = LinearRegressionHistogram.compute_lr_hist(x, y, bins=5)
    assert result["slope"] == pytest.approx(2.0, abs=0.5)
    assert len(result["histogram_counts"]) == 5

def test_lr_hist_basic():
    """Validates perfect linear correlations match explicit algebraic expectations."""
    x = [1, 2, 3, 4]
    y = [3, 5, 7, 9] # Exact formula: y = 2x + 1
    
    result = LinearRegressionHistogram.compute_lr_hist(x, y, bins=2)
    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(1.0)

def test_lr_hist_random():
    """Ensures input shape mismatches throw strict production value errors."""
    with pytest.raises(ValueError):
        LinearRegressionHistogram.compute_lr_hist([1, 2], [1, 2, 3])