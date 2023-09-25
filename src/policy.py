import numpy as np
from utils import phi_sa, phi_a, u1, u2

class TS_variant(object):
    def __init__(self, K, D, lam=1):
        '''
        Initializes the Thompson Sampling variant policy learner
        :param K: number of outcomes
        :param D: number of dimensions in a sample
        :param lam: prior (usually set to 1)
        '''
        self.K = K
        self.D = D
        self.lam = lam

        self.Sigma = np.eye(D) * lam
        self.Sigma_inv = np.eye(D) * (1. / lam)
        self.Omega = np.zeros((D, D))
        self.Beta = np.zeros((K, D))
        self.theta = np.zeros((K, D))
        self.utilities = [u1, u2]

    def update_individual(self, phi, y):
        '''
        Call this function after observing every user.
        :param phi: phi \in R^d, the representation of the sample
        :param y: y \in R^k, the outcome
        '''
        K = self.K
        for k in range(K):
            self.Beta[k] += (phi * y[k])
        self.Omega += np.outer(phi, phi)

    def update_time(self):
        '''
        Call this function after every timestep (in the horizon)
        '''
        self.Sigma += self.Omega
        self.Omega *= 0
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.theta = (self.Sigma_inv @ self.Beta.T).T

    def choose_od_hybrid(self, phi):
        '''
        Choose an action according the optimal design principles.
        :param phi: a list of \phi(s,a) for all the action a
        :return: optimal action
        '''
        A = phi.shape[0]
        scores = []
        for a in range(A):
            score = (phi[a].T @ self.Sigma_inv) @ phi[a]
            scores.append(score)
        scores = np.array(scores)
        choice = np.random.choice(np.flatnonzero(scores == scores.max()))
        return choice

    def choose_mle(self, phi, w, group, choose_step):
        '''
        Choose maximum likelihood estimate
        :param phi:
        :param w:
        :param group:
        :param choose_step:
        :return:
        '''
        A = phi.shape[0]
        scores = []
        for a in range(A):
            # print("Temp Phi: ", temp_phi_a)
            y_it = (phi[a] @ self.theta.squeeze()).item() # Still chooosing noise here.
            if choose_step:
                scores.append(y_it)
            else:
                score = 0 # Only select by utilities.
                for k in range(len(self.utilities)):
                    score += w[k] * self.utilities[k](y_it, phi[a], group) # Use the actual phi[a] here, so we can keep track of notifications
                scores.append(score)
        scores = np.array(scores)
        return np.random.choice(np.flatnonzero(scores == scores.max()))

    def choose_ts(self, phi, w, choose_step, group, print_scores=False):
        '''
        Choose action based on Thompson Sampling
        :param phi:
        :param w:
        :param choose_step:
        :param group:
        :param print_scores:
        :return:
        '''
        A = phi.shape[0]
        scores = []
        theta_tilde = np.random.multivariate_normal(self.theta.squeeze(), (1. / self.lam) * self.Sigma_inv)
        for a in range(A):
            y_it = phi[a] @ theta_tilde
            if choose_step:
                scores.append(y_it) # Choosing action based on what maximizes y_it
            else:
                score = 0
                for k in range(len(self.utilities)):
                    score += w[k] * self.utilities[k](y_it, phi[a], group)
                scores.append(score) # Choosing action based on what maximizes weighted sum of utilies
        scores = np.array(scores)
        if print_scores:
            print(str(scores) + " " + str(np.random.choice(np.flatnonzero(scores == scores.max()))))
        return np.random.choice(np.flatnonzero(scores == scores.max()))
