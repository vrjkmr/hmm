# -*- coding: utf-8 -*-
import numpy as np


class HMM:

    """
    A Hidden Markov Model (HMM).

    ...

    Attributes
    ----------
    Q : numpy.ndarray
        set of N states
    V : numpy.ndarray
        set of M observations
    A : numpy.ndarray
        transition probability matrix of shape N x N
    B : numpy.ndarray
        emission probability matrix of shape N x M
    initial : numpy.ndarray
        initial probability distribution of size N over states

    """

    def __init__(self, Q, V, A, B, initial):
        """Construct a Hidden Markov Model (HMM)."""
        self.Q = Q
        self.V = V
        self.A = A
        self.B = B
        self.N = B.shape[0]
        self.M = B.shape[1]
        self.initial = initial

    def likelihood(self, obs):
        """Compute the likelihood of an observation sequence."""
        prob, _ = self.likelihood_forward(obs)
        return prob

    def likelihood_forward(self, obs):
        """Compute observation likelihood using the forward algorithm."""
        T = len(obs)
        alpha = np.zeros((self.N, T))
        # initialization
        o_0 = self._get_observation_idx(obs[0])
        alpha[:, 0] = self.initial * self.B[:, o_0]
        # recursion
        for t in range(1, T):
            o_t = self._get_observation_idx(obs[t])
            alpha[:, t] = alpha[:, t-1].dot(self.A) * self.B[:, o_t]
        # termination
        prob = alpha[:, T-1].sum()
        return prob, alpha

    def likelihood_backward(self, obs):
        """Compute observation likelihood using the backward algorithm."""
        T = len(obs)
        beta = np.zeros((self.N, T))
        # initialization
        beta[:, T-1] = 1
        # recursion
        for t in range(T-2, -1, -1):
            o_t1 = self._get_observation_idx(obs[t+1])
            beta[:, t] = self.A.dot(self.B[:, o_t1] * beta[:, t+1])
        # termination
        o_0 = self._get_observation_idx(obs[0])
        prob = self.initial.dot(self.B[:, o_0] * beta[:, 0])
        return prob, beta

    def decode(self, obs):
        """Determine the best hidden sequence using the Viterbi algorithm."""
        T = len(obs)
        delta = np.zeros((self.N, T))
        # initialization
        o_0 = self._get_observation_idx(obs[0])
        delta[:, 0] = self.initial * self.B[:, o_0]
        # recursion
        for t in range(1, T):
            o_t = self._get_observation_idx(obs[t])
            delta_prev = delta[:, t-1].reshape(-1, 1)
            delta[:, t] = (delta_prev * self.A).max(axis=0) * self.B[:, o_t]
        # termination
        best_path = self.Q[delta.argmax(axis=0)]
        prob = delta[:, T-1].max()
        return best_path, prob, delta

    def _get_observation_idx(self, o):
        """Get the index value of an observation."""
        return np.argwhere(self.V == o)[0, 0]
