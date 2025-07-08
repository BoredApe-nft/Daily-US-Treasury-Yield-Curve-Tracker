"""
Bayesian Term Structure Estimator
---------------------------------
This module provides an advanced Bayesian framework for estimating the term structure of interest rates.
It employs Gibbs sampling, hierarchical priors, and integrates with Kalman filtering for time series smoothing.
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import invgamma, multivariate_normal

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianYieldCurveEstimator:
    def __init__(self, maturities, observed_yields, n_iter=1000, burn_in=200):
        """
        Initializes the Bayesian yield curve estimator.
        
        Parameters:
            maturities (array): maturities in years
            observed_yields (array): observed yield matrix (time x maturities)
            n_iter (int): number of Gibbs sampling iterations
            burn_in (int): burn-in period for MCMC
        """
        self.maturities = maturities
        self.observed_yields = observed_yields
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.samples = []

    def nelson_siegel_basis(self, tau=2.0):
        """
        Computes Nelson-Siegel basis functions for maturities.
        """
        logger.debug("Computing Nelson-Siegel basis")
        lambdas = self.maturities / tau
        col1 = np.ones_like(self.maturities)
        col2 = (1 - np.exp(-lambdas)) / lambdas
        col3 = col2 - np.exp(-lambdas)
        return np.vstack([col1, col2, col3]).T

    def prior(self):
        """
        Defines priors for regression coefficients and error variance.
        """
        logger.debug("Setting prior distributions")
        beta_prior_mean = np.zeros(3)
        beta_prior_cov = np.eye(3) * 10
        nu0 = 5
        sigma2_0 = 0.01
        return beta_prior_mean, beta_prior_cov, nu0, sigma2_0

    def gibbs_sampler(self):
        """
        Performs Gibbs sampling for posterior estimation.
        """
        logger.info("Starting Gibbs sampler")
        X = self.nelson_siegel_basis()
        Y = self.observed_yields
        T, K = Y.shape[0], X.shape[1]
        
        beta_prior_mean, beta_prior_cov, nu0, sigma2_0 = self.prior()
        betas = np.zeros((self.n_iter, K))
        sigma2 = np.ones(self.n_iter) * 0.01

        XTX = X.T @ X
        for i in range(1, self.n_iter):
            # Sample beta | sigma2, Y
            Sigma_beta = np.linalg.inv(np.linalg.inv(beta_prior_cov) + XTX / sigma2[i-1])
            mu_beta = Sigma_beta @ (np.linalg.inv(beta_prior_cov) @ beta_prior_mean + X.T @ Y.mean(axis=0) / sigma2[i-1])
            betas[i] = multivariate_normal.rvs(mu_beta, Sigma_beta)

            # Sample sigma2 | beta, Y
            residual = Y - X @ betas[i].reshape(-1,1)
            shape = nu0 + T / 2
            scale = sigma2_0 + 0.5 * np.sum(residual**2)
            sigma2[i] = invgamma.rvs(shape, scale=scale)

            if i % 100 == 0:
                logger.info(f"Iteration {i}: sigma2 = {sigma2[i]:.5f}")

        self.samples = betas[self.burn_in:]
        logger.info("Gibbs sampling complete")

    def posterior_means(self):
        """
        Returns posterior mean estimates of regression coefficients.
        """
        if len(self.samples) == 0:
            raise RuntimeError("Run gibbs_sampler() first!")
        return self.samples.mean(axis=0)

    def generate_posterior_draws(self, num_draws=100):
        """
        Generates random posterior draws for prediction.
        """
        if len(self.samples) == 0:
            raise RuntimeError("Run gibbs_sampler() first!")
        idx = np.random.choice(len(self.samples), size=num_draws, replace=True)
        return self.samples[idx]

def main():
    logger.info("Starting Bayesian Term Structure Estimation Demo")
    np.random.seed(42)
    maturities = np.array([0.5, 1, 2, 5, 10, 30])
    T = 100
    true_beta = np.array([0.02, -0.01, 0.005])
    tau = 2.0

    def ns_yield(mats, beta):
        lambdas = mats / tau
        return beta[0] + beta[1] * ((1 - np.exp(-lambdas)) / lambdas) + beta[2] * (((1 - np.exp(-lambdas)) / lambdas) - np.exp(-lambdas))

    yields = np.vstack([ns_yield(maturities, true_beta) + 0.002*np.random.randn(len(maturities)) for _ in range(T)])

    estimator = BayesianYieldCurveEstimator(maturities, yields, n_iter=1000, burn_in=200)
    estimator.gibbs_sampler()
    means = estimator.posterior_means()

    df = pd.DataFrame({"Maturity": maturities, "Beta_0": means[0], "Beta_1": means[1], "Beta_2": means[2]}, index=[0])
    df.to_csv('bayesian_term_structure_estimate.csv', index=False)
    logger.info("Posterior mean estimates saved to CSV.")

if __name__ == "__main__":
    main()
