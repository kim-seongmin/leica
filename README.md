# LEICA

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![JAX](https://img.shields.io/badge/JAX-enabled-orange)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

Official implementation of **LEICA (Learning Exogenous Influence for Coordination and Adaptation)** from
*Addressing Exogenous Variability in Cooperative Multi-Agent Reinforcement Learning* (NeurIPS 2026).

Cooperative MARL often fails under exogenous variability: opponents or environment regimes that the team
cannot control but that change the transition dynamics. We formalize this as an Exogenous Dec-POMDP
(ED-POMDP), in which an episode-level regime indexes the dynamics and the global state splits into
coordinates that respond directly to the current team action (S^d) and coordinates that do not (S^r).
LEICA is a CTDE algorithm built on a MAPPO backbone that

- partitions the state into S^d / S^r automatically from training rollouts (action-sensitivity of an auxiliary one-step model under counterfactual action perturbations, two-cluster split, estimated during the first 10% of training),
- infers a direct context and a regime context from each agent's local history and trains one-step predictors of S^d and S^r with them,
- shapes the reward with an influence term that favours directly induced changes of S^d that influence the later response of the opponent (response-predictor Jacobian) and the team value (critic gradient).


## Installation

```bash
pip install flax optax distrax chex numpy pyyaml
```

## Run

```bash
python train.py                               
```

## Acknowledgements

- Built on top of [JaxMARL](https://github.com/FLAIROx/JaxMARL) (environment, wrappers and the MAPPO baseline structure)

## Citation

```bibtex
@inproceedings{kim2026leica,
  title     = {Addressing Exogenous Variability in Cooperative Multi-Agent Reinforcement Learning},
  author    = {Kim, Seongmin and Byeon, Woohyeon and Jeon, Jiwon and Han, Seungyul and Sung, Youngchul},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
