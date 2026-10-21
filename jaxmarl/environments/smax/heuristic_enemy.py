import jax.numpy as jnp
import jax
from flax.struct import dataclass
import chex
from functools import partial


@dataclass
class HeuristicPolicyState:
    enemy_policy_index: int
    default_target: chex.Array  # the place we are headed for
    last_attacked_enemy: int  # needed to remember where we attacked last

    def __eq__(self, other):
        return jnp.all(other.default_target == self.default_target) & (
            other.last_attacked_enemy == self.last_attacked_enemy
        )


def create_offensive_policy(
    env, team: int, shoot: bool = True, attack_mode: str = "closest", persistency: bool = True, move_mode: str = "center", target_mode: str = "weakest"
):
    num_unit_features = len(env.unit_features)
    num_move_actions = env.num_movement_actions

    def get_heuristic_action(
        key: jax.random.PRNGKey, state: HeuristicPolicyState, obs: chex.Array
    ):
        unit_type = jnp.nonzero(obs[-env.unit_type_bits :], size=1, fill_value=None)[0][0]
        initial_state = get_heuristic_policy_initial_state()
        is_initial_state = initial_state == state
        teams = {0: env.num_allies, 1: env.num_enemies}
        team_size = teams[team]
        other_team_size = teams[1 - team]
        total_units = env.num_allies + env.num_enemies
        attack_range = env.unit_type_attack_ranges[unit_type]
        first_enemy_idx = (team_size - 1) * num_unit_features
        own_feats_idx = (total_units - 1) * num_unit_features
        

        def scaled_position_to_map(position, x_scale, y_scale):
            return position * jnp.array([x_scale, y_scale])

        own_position = scaled_position_to_map(
            obs[own_feats_idx + 1 : own_feats_idx + 3],
            env.map_width,
            env.map_height,
        )
        enemy_positions = jnp.zeros((other_team_size, 2))
        enemy_positions = enemy_positions.at[:, 0].set(
            obs[first_enemy_idx + 1 : own_feats_idx : num_unit_features],
        )
        enemy_positions = enemy_positions.at[:, 1].set(
            obs[first_enemy_idx + 2 : own_feats_idx : num_unit_features]
        )
        enemy_positions = scaled_position_to_map(
            enemy_positions,
            env.unit_type_sight_ranges[unit_type],
            env.unit_type_sight_ranges[unit_type],
        )

        # visible if health is > 0. Otherwise out of range or dead
        enemy_health = obs[first_enemy_idx:own_feats_idx:num_unit_features]
        visible_enemy_mask = enemy_health > 0
        enemy_health = jnp.where(
            visible_enemy_mask,
            enemy_health,
            1e10
        )
        shootable_enemy_mask = (
            jnp.linalg.norm(enemy_positions, axis=-1) < attack_range
        ) & visible_enemy_mask
        can_shoot = jnp.any(shootable_enemy_mask)
        key, key_attack = jax.random.split(key)
        random_attack_action = jax.random.choice(
            key_attack,
            jnp.arange(num_move_actions, other_team_size + num_move_actions),
            p=(shootable_enemy_mask / jnp.sum(shootable_enemy_mask)),
        )
        enemy_dist_unmasked = jnp.linalg.norm(enemy_positions, axis=-1)
        enemy_dist = jnp.where(
            shootable_enemy_mask,
            enemy_dist_unmasked,
            jnp.linalg.norm(jnp.array([env.map_width, env.map_height])),
        )
        enemy_dist_unmasked = jnp.where(
            visible_enemy_mask,
            enemy_dist_unmasked,
            jnp.linalg.norm(jnp.array([env.map_width, env.map_height]))
        )
        enemy_hp = jnp.where(
            shootable_enemy_mask,
            enemy_health,
            1e10,
        )
        closest_attack_action = jnp.argmin(enemy_dist)
        closest_attack_action += num_move_actions
        weakest_attack_action = jnp.argmin(enemy_hp)
        weakest_attack_action += num_move_actions
        new_attack_action = jax.lax.select(
            attack_mode == "random", random_attack_action, closest_attack_action
        )
        new_attack_action = jax.lax.select(
            attack_mode == "weakest", weakest_attack_action, new_attack_action
        )
        # Want to keep attacking the same enemy until it is dead.
        attack_action = jax.lax.select(
            (state.last_attacked_enemy != -1)
            & shootable_enemy_mask[state.last_attacked_enemy]
            & persistency,
            state.last_attacked_enemy + num_move_actions,
            new_attack_action,
        )
        attacked_idx = attack_action - num_move_actions
        state = state.replace(
            last_attacked_enemy=jax.lax.select(
                shootable_enemy_mask[attacked_idx], attacked_idx, -1
            )
        )
        # compute the correct movement action.
        random_enemy_target = jax.random.choice(
            key,
            enemy_positions + own_position,
            p=(visible_enemy_mask / jnp.sum(visible_enemy_mask)),
        )
        closest_enemy_target = enemy_positions[jnp.argmin(enemy_dist_unmasked)] + own_position
        weakest_enemy_target = enemy_positions[jnp.argmin(enemy_health)] + own_position
        enemy_target = jax.lax.select(
            target_mode == "weakest", weakest_enemy_target, random_enemy_target
        )
        enemy_target = jax.lax.select(
            target_mode == "closest", closest_enemy_target, enemy_target
        )
        can_see = jnp.any(visible_enemy_mask)

        target_noise = jnp.array([jax.random.uniform(key, minval=-env.map_width / 20, maxval=env.map_width / 20),
                                  jax.random.uniform(key, minval=-env.map_height / 20, maxval=env.map_height / 20)])

        if move_mode == "ally":
            ally_positions = jnp.zeros((team_size, 2))
            ally_positions = ally_positions.at[:, 0].set(
                obs[: team_size * num_unit_features : num_unit_features + 1]
            )
            ally_positions = ally_positions.at[:, 1].set(
                obs[1 : team_size * num_unit_features : num_unit_features + 1]
            )
            ally_positions = scaled_position_to_map(
                ally_positions,
                env.map_width,
                env.map_height,
            )
            team_center = jnp.mean(ally_positions, axis=0)
            move_target = (team_center + own_position) / 2
        
        else: #center
            move_target = jnp.array([env.map_width / 2, env.map_height / 2])
        
        move_target = move_target + target_noise

        default_target = jax.lax.select(
            is_initial_state | ~can_see,
            jnp.array([[-1, 0], [0, -1]]) @ (own_position - move_target) + move_target,
            state.default_target,
        )
        state = state.replace(default_target=default_target)
        target = jax.lax.cond(
            can_see, lambda: enemy_target, lambda: state.default_target
        )
        vector_to_target = target - own_position
        action_vectors = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        similarity = jnp.dot(action_vectors, vector_to_target)
        move_action = jnp.argmax(similarity)

        if move_mode == "random":
            move_action = jax.random.choice(
                key,
                jnp.arange(num_move_actions),
                p=None,
            )
        return (
            jax.lax.cond(can_shoot & shoot, lambda: attack_action, lambda: move_action),
            state,
        )

    return get_heuristic_action



def get_heuristic_policy_initial_state(enemy_policy_index: int=0):
    return HeuristicPolicyState(
        enemy_policy_index=enemy_policy_index, default_target=jnp.array([0.0, 0.0]), last_attacked_enemy=-1
    )
