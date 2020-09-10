# Hidden Markov Models

This repository contains the code for a Hidden Markov Model (HMM) built from scratch (using Numpy). The code addresses the three fundamental problems of HMMs: **likelihood**, **decoding**, and **learning**.

### Usage

The `HiddenMarkovModel` class is defined in `hmm.py`. To instantiate it, simply define the state space, the observation space, and initialized model parameters (transition probabilities, emission probabilities, and initial probabilities).

```python
from hmm import HiddenMarkovModel

states = ["hot", "cold"]  # weather
observations = [1, 2, 3]  # number of ice creams eaten
initial_probs = [0.8, 0.2]
transition_probs = [[0.6, 0.4], [0.5, 0.5]]
emission_probs = [[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]]

model = HiddenMarkovModel(states, observations, transition_probs,
                          emission_probs, initial_probs)
```

Once instantiated, the three HMM problems can be addressed using three simple function calls.

```python
observation_sequence = [1, 2, 3, 2, 2, 1, 2]

# likelihood : compute likelihood of an observation sequence
prob = model.likelihood(observation_sequence)

# decoding : find the most likely hidden state sequence for an observation sequence
path, prob = model.decode(observation_sequence)

# learning : learn HMM parameters given an observation sequence
model.learn(observation_sequence, iterations=1)
```

### Project structure

This project is organised as follows.

```
.
├── Hidden Markov Model.ipynb           # notebook to display HMM features
├── hmm.py                              # script containing the core HMM class
└── README.md
```

### References

- [Hidden Markov Models](https://web.stanford.edu/~jurafsky/slp3/A.pdf) by Daniel Jurafsky & James H. Martin
- YouTube playlist on [HMMs](https://www.youtube.com/watch?v=J_y5hx_ySCg&list=PLix7MmR3doRo3NGNzrq48FItR3TDyuLCo&ab_channel=djp3) by Prof. Donald J. Patterson
