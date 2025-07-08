import numpy as np
import matplotlib.pyplot as plt

def simulate_garch(T=500, omega=0.0001, alpha=0.05, beta=0.9):
    np.random.seed(42)
    eps = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        eps[t] = np.random.randn() * np.sqrt(sigma2[t])

    return eps, sigma2

def plot_garch_series(eps, sigma2):
    plt.figure(figsize=(10,4))
    plt.subplot(2,1,1)
    plt.plot(eps)
    plt.title('Simulated GARCH Returns')
    plt.subplot(2,1,2)
    plt.plot(sigma2)
    plt.title('Simulated Conditional Variance')
    plt.tight_layout()
    plt.savefig('garch_simulation.png')
    plt.close()

if __name__ == "__main__":
    eps, sigma2 = simulate_garch()
    plot_garch_series(eps, sigma2)
