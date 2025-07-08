import numpy as np
import pandas as pd

class MultiFactorModel:
    def __init__(self, factors, betas, noise_std=0.01):
        self.factors = factors
        self.betas = betas
        self.noise_std = noise_std

    def generate_returns(self):
        n_periods, n_factors = self.factors.shape
        n_assets = self.betas.shape[0]
        returns = self.factors @ self.betas.T + np.random.randn(n_periods, n_assets) * self.noise_std
        return returns

def main():
    np.random.seed(0)
    n_assets = 5
    n_factors = 3
    n_periods = 250

    factors = np.random.randn(n_periods, n_factors)
    betas = np.random.randn(n_assets, n_factors)

    model = MultiFactorModel(factors, betas)
    returns = model.generate_returns()

    df = pd.DataFrame(returns, columns=[f'Asset_{i+1}' for i in range(n_assets)])
    df.to_csv('simulated_asset_returns.csv', index=False)
    print('Simulated multi-factor returns saved.')

if __name__ == "__main__":
    main()
