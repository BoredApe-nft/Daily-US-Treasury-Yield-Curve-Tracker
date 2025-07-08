import numpy as np
import pandas as pd

class OptimalPortfolioAllocator:
    def __init__(self, expected_returns, covariance_matrix, risk_aversion=3.0):
        self.mu = expected_returns
        self.Sigma = covariance_matrix
        self.risk_aversion = risk_aversion

    def solve_markowitz(self):
        ones = np.ones(len(self.mu))
        inv_Sigma = np.linalg.inv(self.Sigma)
        weights = inv_Sigma @ ones
        weights /= ones.T @ inv_Sigma @ ones
        return weights

    def compute_optimal_allocation(self):
        inv_Sigma = np.linalg.inv(self.Sigma)
        return (1/self.risk_aversion) * inv_Sigma @ self.mu

if __name__ == "__main__":
    np.random.seed(42)
    assets = ['Bond1', 'Bond2', 'Bond3', 'Bond4']
    expected_returns = np.random.rand(4) * 0.05
    covariance_matrix = np.random.rand(4,4)
    covariance_matrix = covariance_matrix @ covariance_matrix.T

    allocator = OptimalPortfolioAllocator(expected_returns, covariance_matrix)
    w = allocator.compute_optimal_allocation()

    df = pd.DataFrame({'Asset': assets, 'Weight': w})
    df.to_csv('optimal_portfolio_weights.csv', index=False)
    print(df)
