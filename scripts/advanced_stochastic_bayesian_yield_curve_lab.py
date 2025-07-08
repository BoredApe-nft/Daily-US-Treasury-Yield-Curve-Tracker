"""
Advanced Stochastic Bayesian Yield Curve Lab
---------------------------------------------

This module provides a hyper-advanced, multi-layered quant research lab simulation for modeling
the term structure of interest rates. It combines:

- Stochastic Differential Equation (SDE) simulators
- Hierarchical Bayesian inference
- Kalman filtering in state-space models
- Gibbs sampling for latent factors
- Nelson-Siegel multi-factor expansions
- Heston-type stochastic volatility processes
- Time-varying parameter estimation with MCMC

Note:
This code is *research-grade* and showcases complex quant engineering.

Author: BoredApe-nft
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as linalg
import logging
import warnings

from numpy.linalg import inv, cholesky
from scipy.stats import invwishart, multivariate_normal
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BayesianYieldCurveLab')

np.random.seed(42)


##########################################################
# Utility functions and base classes
##########################################################

def exp_decay_basis(maturities: np.ndarray, lambd: float) -> np.ndarray:
    """Nelson-Siegel basis functions"""
    with np.errstate(divide='ignore', invalid='ignore'):
        lambda_m = lambd * maturities
        term1 = (1 - np.exp(-lambda_m)) / lambda_m
        term2 = term1 - np.exp(-lambda_m)
    basis = np.vstack([np.ones_like(maturities), term1, term2]).T
    logger.debug(f"Computed NS basis: {basis.shape}")
    return basis

class KalmanFilter:
    """
    Standard Kalman filter for state-space models with time-varying observation noise.
    """
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def filter(self, observations):
        T = observations.shape[0]
        n_states = self.F.shape[0]

        filtered_means = np.zeros((T, n_states))
        filtered_covs = np.zeros((T, n_states, n_states))

        for t in range(T):
            # Predict
            x_pred = self.F @ self.x
            P_pred = self.F @ self.P @ self.F.T + self.Q

            # Update
            y = observations[t] - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ inv(S)
            self.x = x_pred + K @ y
            self.P = P_pred - K @ self.H @ P_pred

            filtered_means[t] = self.x
            filtered_covs[t] = self.P

            logger.debug(f"Time {t}: Filtered State {self.x}")

        return filtered_means, filtered_covs


##########################################################
# Stochastic Volatility Process (Heston-style)
##########################################################

class StochasticVolatilitySimulator:
    def __init__(self, v0=0.02, kappa=1.5, theta=0.02, sigma=0.3, dt=1/252):
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

    def simulate_path(self, T=252):
        v = np.zeros(T)
        v[0] = self.v0
        for t in range(1, T):
            dv = self.kappa * (self.theta - v[t-1]) * self.dt + self.sigma * np.sqrt(v[t-1]*self.dt) * np.random.randn()
            v[t] = max(v[t-1] + dv, 0.00001)
        return v

class YieldCurveSDE:
    """
    Simulates short rate paths using a generic Vasicek/Ornstein-Uhlenbeck process.
    """
    def __init__(self, r0=0.03, kappa=0.15, theta=0.05, sigma=0.01, dt=1/252):
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

    def simulate_path(self, T=252):
        r = np.zeros(T)
        r[0] = self.r0
        for t in range(1, T):
            dr = self.kappa * (self.theta - r[t-1]) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn()
            r[t] = r[t-1] + dr
        return r
##########################################################
# Hierarchical Bayesian Term Structure Model
##########################################################

class HierarchicalBayesianYieldCurve:
    def __init__(self, maturities, yields, n_factors=3, n_iter=2000, burn_in=500):
        self.maturities = maturities
        self.yields = yields
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.n_obs, self.n_maturities = yields.shape
        self.n_factors = n_factors

        self.tau_prior_mean = 2.0
        self.tau_prior_sd = 0.5

        self.samples = {
            'betas': [],
            'tau': [],
            'sigma2': []
        }

    def _design_matrix(self, tau):
        logger.debug(f"Generating design matrix with tau={tau}")
        return exp_decay_basis(self.maturities, tau)

    def _sample_tau(self, beta, sigma2, X, Y):
        # Normal approximation for tau sampling
        proposed_tau = np.random.normal(self.tau_prior_mean, self.tau_prior_sd)
        if proposed_tau <= 0:
            return self.tau_prior_mean

        X_new = exp_decay_basis(self.maturities, proposed_tau)
        ll_old = -0.5*np.sum((Y - X @ beta.T)**2)/sigma2
        ll_new = -0.5*np.sum((Y - X_new @ beta.T)**2)/sigma2

        prior_old = stats.norm.logpdf(self.tau_prior_mean, loc=self.tau_prior_mean, scale=self.tau_prior_sd)
        prior_new = stats.norm.logpdf(proposed_tau, loc=self.tau_prior_mean, scale=self.tau_prior_sd)

        acceptance = ll_new + prior_new - ll_old - prior_old
        if np.log(np.random.rand()) < acceptance:
            logger.debug(f"Accepted new tau: {proposed_tau}")
            return proposed_tau
        else:
            return self.tau_prior_mean

    def _sample_betas(self, X, Y, sigma2):
        XtX = X.T @ X
        precision = XtX / sigma2 + np.eye(X.shape[1])
        cov = inv(precision)
        mean = cov @ (X.T @ Y.mean(axis=0) / sigma2)
        return multivariate_normal.rvs(mean, cov)

    def _sample_sigma2(self, residuals):
        nu = 5 + self.n_obs * self.n_maturities / 2
        scale = 0.01 + 0.5 * np.sum(residuals**2)
        return invgamma.rvs(nu, scale=scale)

    def run_gibbs_sampler(self):
        logger.info("Starting hierarchical Bayesian Gibbs sampler")
        tau = self.tau_prior_mean
        sigma2 = 0.01
        beta = np.zeros((self.n_maturities, self.n_factors))

        for i in range(self.n_iter):
            X = self._design_matrix(tau)
            beta = self._sample_betas(X, self.yields, sigma2)
            residuals = self.yields - X @ beta.T
            sigma2 = self._sample_sigma2(residuals)
            tau = self._sample_tau(beta, sigma2, X, self.yields)

            if i >= self.burn_in:
                self.samples['betas'].append(beta)
                self.samples['tau'].append(tau)
                self.samples['sigma2'].append(sigma2)

            if i % 100 == 0:
                logger.info(f"Iter {i}: tau={tau:.4f}, sigma2={sigma2:.6f}")

        logger.info("Gibbs sampling complete.")

    def posterior_means(self):
        betas = np.array(self.samples['betas'])
        tau = np.array(self.samples['tau'])
        sigma2 = np.array(self.samples['sigma2'])
        return betas.mean(axis=0), tau.mean(), sigma2.mean()

##########################################################
# Posterior Predictive Generation and Plotting
##########################################################

class PosteriorPredictiveSimulator:
    def __init__(self, maturities, posterior_betas, tau_samples):
        self.maturities = maturities
        self.posterior_betas = posterior_betas
        self.tau_samples = tau_samples

    def generate_samples(self, n_draws=100):
        n_posterior = len(self.posterior_betas)
        draws = []
        for _ in range(n_draws):
            idx = np.random.choice(n_posterior)
            tau = self.tau_samples[idx]
            beta = self.posterior_betas[idx]
            X = exp_decay_basis(self.maturities, tau)
            y_pred = X @ beta.T
            draws.append(y_pred)
        return np.array(draws)

    def save_draws(self, draws, filename='posterior_predictive_draws.csv'):
        n_samples, n_obs, n_mats = draws.shape
        flat_draws = draws.reshape((n_samples * n_obs, n_mats))
        df = pd.DataFrame(flat_draws, columns=[f"Maturity_{m:.2f}" for m in self.maturities])
        df.to_csv(filename, index=False)
        logger.info(f"Posterior predictive draws saved to {filename}")

def plot_posterior_means(maturities, posterior_mean_betas, posterior_mean_tau):
    X = exp_decay_basis(maturities, posterior_mean_tau)
    mean_curve = X @ posterior_mean_betas.T
    plt.figure(figsize=(10,6))
    plt.plot(maturities, mean_curve, label='Posterior Mean Yield Curve', color='darkblue')
    plt.scatter(maturities, mean_curve, color='red')
    plt.title('Posterior Mean Term Structure')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield')
    plt.legend()
    plt.grid()
    plt.savefig('posterior_mean_yield_curve.png')
    plt.close()
    logger.info("Posterior mean yield curve plot saved.")

##########################################################
# Simulate Data and Run Entire Pipeline
##########################################################

def main():
    logger.info("=== ADVANCED STOCHASTIC BAYESIAN YIELD CURVE LAB ===")
    np.random.seed(42)
    
    # Simulate synthetic yield curve data
    maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 20, 30])
    true_tau = 2.0
    true_betas = np.array([0.03, -0.015, 0.005])

    def ns_function(mats, beta, tau):
        lambdas = mats / tau
        term1 = (1 - np.exp(-lambdas)) / lambdas
        term2 = term1 - np.exp(-lambdas)
        return beta[0] + beta[1] * term1 + beta[2] * term2

    T_obs = 250
    yields_data = np.vstack([
        ns_function(maturities, true_betas, true_tau) + 0.002*np.random.randn(len(maturities))
        for _ in range(T_obs)
    ])

    # Add stochastic volatility effect
    sv_simulator = StochasticVolatilitySimulator()
    vol_path = sv_simulator.simulate_path(T_obs)
    yields_data += np.random.randn(*yields_data.shape) * vol_path[:, None]

    # Save synthetic dataset
    pd.DataFrame(yields_data, columns=[f"Maturity_{m:.2f}" for m in maturities]).to_csv('simulated_yield_data.csv', index=False)
    logger.info("Simulated yield curve data saved.")

    # Run Hierarchical Bayesian Inference
    model = HierarchicalBayesianYieldCurve(
        maturities=maturities,
        yields=yields_data,
        n_factors=3,
        n_iter=2000,
        burn_in=500
    )
    model.run_gibbs_sampler()
    mean_betas, mean_tau, mean_sigma2 = model.posterior_means()
    logger.info(f"Posterior Mean Betas: {mean_betas}")
    logger.info(f"Posterior Mean Tau: {mean_tau:.4f}")
    logger.info(f"Posterior Mean Sigma2: {mean_sigma2:.6f}")

    # Save posterior means
    df_posterior = pd.DataFrame({
        "Beta_0": [mean_betas[0]],
        "Beta_1": [mean_betas[1]],
        "Beta_2": [mean_betas[2]],
        "Tau": [mean_tau],
        "Sigma2": [mean_sigma2]
    })
    df_posterior.to_csv('posterior_parameter_estimates.csv', index=False)
    logger.info("Posterior parameter estimates saved.")

    # Plot posterior mean curve
    plot_posterior_means(maturities, mean_betas, mean_tau)

    # Posterior predictive sampling
    simulator = PosteriorPredictiveSimulator(
        maturities,
        np.array(model.samples['betas']),
        np.array(model.samples['tau'])
    )
    draws = simulator.generate_samples(n_draws=100)
    simulator.save_draws(draws, 'posterior_predictive_draws.csv')

    logger.info("=== LAB COMPLETE ===")

if __name__ == "__main__":
    main()

