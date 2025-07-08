import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = self.P - K @ self.H @ self.P

def fit_yield_curve_with_kalman(observed_rates):
    n = len(observed_rates)
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.001
    R = np.eye(1) * 0.01
    x0 = np.zeros((2,1))
    P0 = np.eye(2)

    kf = KalmanFilter(F, H, Q, R, x0, P0)
    estimates = []

    for z in observed_rates:
        kf.predict()
        kf.update(np.array([[z]]))
        estimates.append(kf.x.flatten())

    estimates = np.array(estimates)
    np.savetxt('kalman_curve_estimates.csv', estimates, delimiter=',')
    print('Saved Kalman filter estimates.')

if __name__ == "__main__":
    np.random.seed(42)
    observed = 0.05 + 0.005*np.random.randn(100)
    fit_yield_curve_with_kalman(observed)
