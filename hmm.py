# -*- coding: utf-8 -*-
import numpy as np


class HiddenMarkovModel:

    """A Hidden Markov Model (HMM).

    Attributes
    ----------
    states : array_like or numpy ndarray
        List of states.

    observations : array_like or numpy ndarray
        Observations space array.

    tp : array_like or numpy ndarray
        Transition probability matrix which stores probability of
        moving from state i (row) to state j (col).

    ep : array_like or numpy ndarray
        Emission probability matrix which stores probability of
        seeing observation o (col) from state s (row).

    pi : array_like or numpy ndarray
        Initial state probabilities array.

    """

    def __init__(self, states, observations, tp, ep, pi):

        self.states = np.array(states)
        self.observations = np.array(observations)
        self.num_states = self.states.shape[0]
        self.num_observations = self.observations.shape[0]
        self.tp = np.array(tp)
        self.ep = np.array(ep)
        self.pi = np.array(pi)

    def likelihood(self, obs):
        """Compute the likelihood of an observation sequence.

        Parameters
        ----------
        obs : array_like or numpy ndarray
            Sequence of observations.

        Returns
        -------
        prob : float
            Probability likelihood for observation sequence.

        """

        prob, _ = self.likelihood_forward(obs)
        return prob

    def likelihood_forward(self, obs):
        """Compute observation likelihood using the forward algorithm.

        Parameters
        ----------
        obs : array_like or numpy ndarray
            Sequence of observations of size T.

        Returns
        -------
        prob : float
            Probability likelihood for observation sequence.

        alpha : numpy ndarray
            Forward probability matrix of shape (num_states x T).

        """

        T = len(obs)
        alpha = np.zeros((self.num_states, T))

        # initialization
        o_0 = self._get_observation_idx(obs[0])
        alpha[:, 0] = self.pi * self.ep[:, o_0]

        # recursion
        for t in range(1, T):
            o_t = self._get_observation_idx(obs[t])
            alpha[:, t] = alpha[:, t-1].dot(self.tp) * self.ep[:, o_t]

        # termination
        prob = alpha[:, T-1].sum()

        return prob, alpha

    def likelihood_backward(self, obs):
        """Compute observation likelihood using the backward algorithm.

        Parameters
        ----------
        obs : array_like or numpy ndarray
            Sequence of observations of size T.

        Returns
        -------
        prob : float
            Probability likelihood for observation sequence.

        beta : numpy ndarray
            Backward probability matrix of shape (num_states x T).

        """

        T = len(obs)
        beta = np.zeros((self.num_states, T))

        # initialization
        beta[:, T-1] = 1

        # recursion
        for t in range(T-2, -1, -1):
            o_t1 = self._get_observation_idx(obs[t+1])
            beta[:, t] = self.tp.dot(self.ep[:, o_t1] * beta[:, t+1])

        # termination
        o_0 = self._get_observation_idx(obs[0])
        prob = self.pi.dot(self.ep[:, o_0] * beta[:, 0])

        return prob, beta

    def decode(self, obs):
        """Determine the best hidden sequence using the Viterbi algorithm.

        Parameters
        ----------
        obs : array_like or numpy ndarray
            Sequence of observations of size T.

        Returns
        -------
        path : numpy ndarray
            Sequence of states of size T.

        prob : float
            Probability likelihood for observation sequence along path.

        """

        T = len(obs)
        delta = np.zeros((self.num_states, T))

        # initialization
        o_0 = self._get_observation_idx(obs[0])
        delta[:, 0] = self.pi * self.ep[:, o_0]

        # recursion
        for t in range(1, T):
            o_t = self._get_observation_idx(obs[t])
            delta_prev = delta[:, t-1].reshape(-1, 1)
            delta[:, t] = (delta_prev * self.tp).max(axis=0) * self.ep[:, o_t]

        # termination
        path = self.states[delta.argmax(axis=0)]
        prob = delta[:, T-1].max()

        return path, prob

    def learn(self, obs, iterations=1):
        """Learn parameters from an observation sequence using Baum-Welch.

        Parameters
        ----------
        obs : array_like or numpy ndarray
            Sequence of observations of size T.

        iterations : int, optional
            Number of Expectation-Maximization (EM) iterations.
            Defaults to 1.

        """

        for _ in range(iterations):
            T = len(obs)

            # expectation step
            likelihood, alpha = self.likelihood_forward(obs)
            _, beta = self.likelihood_backward(obs)
            gamma = alpha * beta / (alpha * beta).sum(axis=0)
            xi = np.zeros((self.num_states, self.num_states, T-1))
            for t in range(T-1):
                o_t1 = self._get_observation_idx(obs[t+1])
                for i in range(self.num_states):
                    xi[i, :, t] = alpha[i, t] * self.tp[i, :] \
                                    * self.ep[:, o_t1] * beta[:, t+1]
            xi /= xi.sum(axis=(0, 1))

            # maximization step
            self.pi = gamma[:, 0]
            self.tp = xi.sum(axis=2) / gamma[:, :-1].sum(axis=1).reshape(-1, 1)
            for idx, o in enumerate(self.observations):
                indices = np.argwhere(obs == o).flatten()
                self.ep[:, idx] = gamma[:, indices].sum(axis=1) \
                    / gamma.sum(axis=1)

    def _get_observation_idx(self, o):
        """Get the vocabulary index value of an observation."""
        return np.argwhere(self.observations == o).flatten().item()
