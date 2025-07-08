import numpy as np
import matplotlib.pyplot as plt

class VasicekModel:
    def __init__(self, kappa, theta, sigma, r0):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0

    def simulate(self, T, dt=0.01):
        N = int(T/dt)
        rates = np.zeros(N)
        rates[0] = self.r0
        for i in range(1, N):
            dr = self.kappa * (self.theta - rates[i-1]) * dt + self.sigma * np.sqrt(dt) * np.random.randn()
            rates[i] = rates[i-1] + dr
        return rates

def plot_simulated_paths():
    model = VasicekModel(kappa=0.15, theta=0.05, sigma=0.01, r0=0.03)
    paths = [model.simulate(5) for _ in range(10)]
    for path in paths:
        plt.plot(path)
    plt.title('Simulated Short Rate Paths (Vasicek)')
    plt.xlabel('Time Steps')
    plt.ylabel('Short Rate')
    plt.savefig('simulated_vasicek_paths.png')
    plt.close()

if __name__ == "__main__":
    plot_simulated_paths()
