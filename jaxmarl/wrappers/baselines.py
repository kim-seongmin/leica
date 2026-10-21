"""Wrappers used by the LEICA trainer (trimmed from the JaxMARL baselines wrappers)."""
from functools import partial
from typing import List, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Box, Discrete, MultiDiscrete


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats_smax(self, x: dict):
        return jnp.stack([x[a] for a in self._env.ally_agents])


@struct.dataclass
class SMAXLogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    won_episode: int
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_won_episode: int


class SMAXLogWrapper(JaxMARLWrapper):
    """Tracks episode returns, lengths and the win flag of the ally team."""

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = SMAXLogEnvState(
            env_state,
            jnp.zeros((self._env.num_allies,)),
            jnp.zeros((self._env.num_allies,)),
            jnp.zeros((self._env.num_allies,)),
            jnp.zeros((self._env.num_allies,)),
            jnp.zeros((self._env.num_allies,)),
            jnp.zeros((self._env.num_allies,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: SMAXLogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, SMAXLogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        batch_reward = self._batchify_floats_smax(reward)
        new_episode_return = state.episode_returns + self._batchify_floats_smax(reward)
        new_episode_length = state.episode_lengths + 1
        new_won_episode = (batch_reward >= 1.0).astype(jnp.float32)
        state = SMAXLogEnvState(
            env_state=env_state,
            won_episode=new_won_episode * (1 - ep_done),
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
            returned_won_episode=state.returned_won_episode * (1 - ep_done)
            + new_won_episode * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_won_episode"] = state.returned_won_episode
        info["returned_episode"] = jnp.full((self._env.num_allies,), ep_done)
        return obs, state, reward, done, info


def get_space_dim(space):
    # get the proper action/obs space from Discrete-MultiDiscrete-Box spaces
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, (Box, MultiDiscrete)):
        return np.prod(space.shape)
    else:
        raise NotImplementedError('Current wrapper works only with Discrete/MultiDiscrete/Box action and obs spaces')


class SMAXRolloutManager(JaxMARLWrapper):
    """
    Rollout manager for centralised training with parameter sharing.
    - Batchifies multiple environments (the number of parallel envs is defined by batch_size in __init__).
    - Adds a global state (obs["__all__"]) and a global reward (rewards["__all__"]) to the env.step returns.
    - Pads the observations of the agents so that they all have the same length.
    - Adds an agent id (one hot encoded) to the observation vectors.
    """

    def __init__(self, env: MultiAgentEnv, batch_size: int, ally_agents: List = None, preprocess_obs: bool = True):

        super().__init__(env)

        self.batch_size = batch_size
        self.ally_agents = self.agents if ally_agents is None else ally_agents
        self.preprocess_obs = preprocess_obs

        # batched action sampling
        self.batch_samplers = {agent: jax.jit(jax.vmap(self.action_space(agent).sample, in_axes=0)) for agent in self.ally_agents}

        # assumes the observations are flattened vectors
        self.max_obs_length = max(list(map(lambda x: get_space_dim(x), self.observation_spaces.values())))
        self.max_action_space = max(list(map(lambda x: get_space_dim(x), self.action_spaces.values())))
        self.max_agents = max([self.num_allies, self.num_enemies])
        self.obs_size = self.max_obs_length
        if self.preprocess_obs:
            self.obs_size += self.max_agents

        # agents ids
        self.agents_one_hot = {a: oh for a, oh in zip(self.ally_agents, jnp.eye(self.max_agents)[:self.num_allies])}
        # valid actions
        self.valid_actions = {a: jnp.arange(u.n) for a, u in self.action_spaces.items()}
        self.valid_actions_oh = {a: jnp.concatenate((jnp.ones(u.n), jnp.zeros(self.max_action_space - u.n))) for a, u in self.action_spaces.items()}

        self.global_reward = lambda rewards: rewards[self.ally_agents[0]]
        self.get_valid_actions = lambda state: jax.vmap(env.get_avail_actions)(state)

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, key):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_reset, in_axes=0)(keys)

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, key, states, actions):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_step, in_axes=(0, 0, 0))(keys, states, actions)

    @partial(jax.jit, static_argnums=0)
    def wrapped_reset(self, key):
        obs_, state = self._env.reset(key)
        if self.preprocess_obs:
            obs = jax.tree.map(self._preprocess_obs, {agent: obs_[agent] for agent in self.ally_agents}, self.agents_one_hot)
            obs["world_state"] = obs_["world_state"]
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(self, key, state, actions):
        obs_, state, reward, done, infos = self._env.step(key, state, actions)
        if self.preprocess_obs:
            obs = jax.tree.map(self._preprocess_obs, {agent: obs_[agent] for agent in self.ally_agents}, self.agents_one_hot)
            obs = jax.tree.map(lambda d, o: jnp.where(d, 0., o), {agent: done[agent] for agent in self.ally_agents}, obs)  # ensure that the obs are 0s for done agents
            obs["world_state"] = obs_["world_state"]
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        reward["__all__"] = self.global_reward(reward)
        return obs, state, reward, done, infos

    @partial(jax.jit, static_argnums=0)
    def global_state(self, obs, state):
        return jnp.concatenate([obs[agent] for agent in self.ally_agents], axis=-1)

    @partial(jax.jit, static_argnums=0)
    def global_reward(self, reward):
        return jnp.stack([reward[agent] for agent in self.ally_agents]).sum(axis=0)

    def batch_sample(self, key, agent):
        return self.batch_samplers[agent](jax.random.split(key, self.batch_size)).astype(int)

    @partial(jax.jit, static_argnums=0)
    def get_valid_actions(self, state):
        # default is to return the same valid actions one hot encoded for each env
        return {agent: jnp.tile(actions, self.batch_size).reshape(self.batch_size, -1) for agent, actions in self.valid_actions_oh.items()}

    @partial(jax.jit, static_argnums=0)
    def _preprocess_obs(self, arr, extra_features):
        # flatten
        arr = arr.flatten()
        # pad the observation vectors to the maximum length
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, max(0, self.max_obs_length - arr.shape[-1]))]
        arr = jnp.pad(arr, pad_width, mode='constant', constant_values=0)
        # concatenate the extra features
        arr = jnp.concatenate((arr, extra_features), axis=-1)
        return arr
