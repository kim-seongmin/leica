"""SMAX against a scripted opponent whose strategy (the exogenous regime) is drawn per episode.

Each regime is one combination of (shoot, attack mode, persistency, move mode, target mode,
initial distance); see `HeuristicEnemySMAX.tasks` for the enumeration used by the config
`TRAINING_SET` / `TEST_SET` indices.
"""
import itertools
from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.smax.heuristic_enemy import (
    create_offensive_policy,
    get_heuristic_policy_initial_state,
)
from jaxmarl.environments.smax.smax_env import SMAX


@dataclass
class State:
    # underlying jaxmarl env state
    state: ...
    # the enemy policy state. Needed for recurrent policies or
    # remembering details about previous observations for heuristics.
    enemy_policy_state: ...


class EnemySMAX(MultiAgentEnv):
    """Class that presents the SMAX environment as a single-player
    (but still multi-agent) environment. Functions like a wrapper, but
    not linked with any of the wrapper code because that is used differently."""

    def __init__(self, **env_kwargs):
        self._env = SMAX(**env_kwargs)
        # only one team
        self.num_enemies = self._env.num_enemies
        self.num_allies = self._env.num_allies
        self.ally_agents = [f"ally_{i}" for i in range(self.num_allies)]
        self.enemy_agents = [f"enemy_{i}" for i in range(self.num_enemies)]
        self.num_agents = self.num_allies + self.num_enemies
        self.agents = self.ally_agents + self.enemy_agents
        self.observation_spaces = {
            i: self._env.observation_spaces[i] for i in self.ally_agents
        }
        self.action_spaces = {i: self._env.action_spaces[i] for i in self.ally_agents}

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def get_enemy_actions(self, key, enemy_policy_state, enemy_obs):
        raise NotImplementedError

    def get_enemy_policy_initial_state(self, key):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ):
        jaxmarl_state = state.state
        enemy_obs = self._env.get_obs_unit_list(jaxmarl_state)
        enemy_obs = jnp.array([enemy_obs[agent] for agent in self.enemy_agents])
        key, action_key = jax.random.split(key)
        enemy_actions, enemy_policy_state = self.get_enemy_actions(
            action_key, state.enemy_policy_state, enemy_obs
        )
        enemy_actions = jnp.array([enemy_actions[i] for i in self.enemy_agents])
        actions = jnp.array([actions[i] for i in self.ally_agents])
        enemy_movement_actions, enemy_attack_actions = (
            self._env._decode_discrete_actions(enemy_actions)
        )
        if self._env.action_type == "continuous":
            cont_actions = jnp.zeros((len(self.agents), 4))
            cont_actions = cont_actions.at[: self.num_allies].set(actions)
            key, action_key = jax.random.split(key)
            ally_movement_actions, ally_attack_actions = (
                self._env._decode_continuous_actions(
                    action_key, jaxmarl_state, cont_actions
                )
            )
            ally_movement_actions = ally_movement_actions[: self.num_allies]
            ally_attack_actions = ally_attack_actions[: self.num_allies]
        else:
            ally_movement_actions, ally_attack_actions = (
                self._env._decode_discrete_actions(actions)
            )

        movement_actions = jnp.concatenate(
            [ally_movement_actions, enemy_movement_actions], axis=0
        )
        attack_actions = jnp.concatenate([ally_attack_actions, enemy_attack_actions], axis=0)

        obs, jaxmarl_state, rewards, dones, infos = self._env.step_env_no_decode(
            key,
            jaxmarl_state,
            (movement_actions, attack_actions),
            get_state_sequence=False,
        )
        new_obs = {agent: obs[agent] for agent in self.ally_agents}
        new_obs["world_state"] = obs["world_state"]
        rewards = {agent: rewards[agent] for agent in self.ally_agents}
        all_done = dones["__all__"]
        dones = {agent: dones[agent] for agent in self.ally_agents}
        dones["__all__"] = all_done
        dones["ally"] = all_done
        dones["enemy"] = all_done

        state = state.replace(
            enemy_policy_state=enemy_policy_state, state=jaxmarl_state
        )
        return new_obs, state, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State):
        avail_actions = self._env.get_avail_actions(state.state)
        return {agent: avail_actions[agent] for agent in self.ally_agents}

    def get_all_unit_obs(self, state: State):
        return self._env.get_obs(state.state)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        obs = self.get_all_unit_obs(state)
        return {agent: obs[agent] for agent in self.ally_agents}

    def get_world_state(self, state: State):
        return self._env.get_world_state(state.state)

    def is_terminal(self, state: State):
        return self._env.is_terminal(state.state)


class HeuristicEnemySMAX(EnemySMAX):
    """SMAX with a scripted opponent team. At every reset one regime is drawn uniformly from
    `task_ids` (indices into `self.tasks`); the regime fixes the opponent script and the initial
    distance between the two teams for the whole episode."""

    def __init__(self, task_ids=[0], **env_kwargs):
        super().__init__(**env_kwargs)
        enemy_shoots = [True, False]
        attack_mode = ["closest", "weakest", "random"]
        persistency = [True, False]
        move_mode = ["center", "ally", "random"]
        target_mode = ["closest", "weakest", "random"]
        distances = [0.4, 0.6, 0.8]
        self.tasks = [combination for combination in itertools.product(
            enemy_shoots, attack_mode, persistency, move_mode, target_mode, distances)]
        self.task_ids = jnp.array(task_ids)
        self.policy_set = [create_offensive_policy(self._env, 1, *self.tasks[task][:-1]) for task in task_ids]
        self.dist_set = jnp.array([self.tasks[task][-1] for task in task_ids])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        key, reset_key = jax.random.split(key)
        enemy_policy_index = jax.random.choice(reset_key, len(self.task_ids))
        dist = self.dist_set[enemy_policy_index]
        obs, state = self._env.reset(reset_key, dist)
        enemy_policy_state = self.get_enemy_policy_initial_state(key, enemy_policy_index)
        new_obs = {agent: obs[agent] for agent in self.ally_agents}
        new_obs["world_state"] = obs["world_state"]
        return new_obs, State(state=state, enemy_policy_state=enemy_policy_state)

    def get_enemy_policy_initial_state(self, key, enemy_policy_index):
        state = jax.tree.map(
            lambda *xs: jnp.stack(xs),
            *([get_heuristic_policy_initial_state(enemy_policy_index)] * self.num_enemies),
        )
        return state

    def get_enemy_actions(self, key, policy_state, enemy_obs):
        heuristic_action_key = jax.random.split(key, num=self.num_enemies)
        index = policy_state.enemy_policy_index

        def apply_policy_per_enemy(k, s, o, i):
            branches = [lambda k, s, o, fn=fn: fn(k, s, o) for fn in self.policy_set]
            return jax.lax.switch(i, branches, k, s, o)
        enemy_actions, policy_state = jax.vmap(apply_policy_per_enemy)(
            heuristic_action_key, policy_state, enemy_obs, index
        )
        enemy_actions = {
            agent: enemy_actions[self._env.agent_ids[agent] - self.num_agents]
            for agent in self.enemy_agents
        }
        return enemy_actions, policy_state
