import numpy as np
'''
class OrnsteinUhlenbeckActionNoise:
    def __init__(self,mu,sigma,theta=0.15,dt=1e-2,x0=None):
        self.theta = theta
        self.mu=mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
'''