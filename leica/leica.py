"""LEICA: Learning Exogenous Influence for Coordination and Adaptation.

A CTDE actor-critic (MAPPO backbone, parameter sharing, recurrent actor/critic) extended with
  * two history-conditioned latent contexts per agent: a direct context z (named z_int in the
    code) and a regime context y (z_ext),
  * two one-step predictors trained by a VariBAD-style ELBO: the direct head predicts the next
    S^d from (S, a^i, z) and the response head predicts the next S^r from (S, y),
  * an influence reward: the weighted magnitude of the counterfactual change of S^d caused by
    the agent's action, with coordinate weights built from the response-predictor Jacobian
    (mediated influence) and the critic gradient (value sensitivity),
  * an automatic partition of the world-state coordinates into S^d / S^r: an auxiliary one-step
    model f(S, A) is trained, every coordinate gets an action-sensitivity score (its normalised
    predicted change under counterfactual single-agent action perturbations) and a 1-D two-means
    split assigns the high-score cluster to S^d. The partition is estimated at the end of the
    first PARTITION_WARMUP_FRAC of training and then kept fixed; the influence reward starts
    once the partition is available.

Shapes: T = NUM_STEPS, N = number of allies, E = NUM_ENVS, A = action dim, S = world-state dim
(world state + agent one-hot), d = number of dynamic coordinates (hp, x, y, cooldown per unit).
"""
from functools import partial
from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import JaxMARLWrapper, SMAXLogWrapper, SMAXRolloutManager
from leica.networks import MLP, ActorDRNN, CriticDRNN, InfluenceDecoder, ScannedRNN, VAEncoder


class SMAXWorldStateWrapper(JaxMARLWrapper):
    """Provides a `"world_state"` observation for the centralised critic:
    (num_allies, world_state_size), the world state replicated per agent plus the agent one-hot."""

    def __init__(self, env: HeuristicEnemySMAX, obs_with_agent_id=True):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id

        if not self.obs_with_agent_id:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state_fn(obs, state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state

    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)

    def world_state_size(self):
        return self._world_state_size


class Transition(NamedTuple):
    global_done: jnp.ndarray
    obs: jnp.ndarray
    state: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    dones: jnp.ndarray
    avail_actions: jnp.ndarray
    r_indiv: jnp.ndarray
    r_team: jnp.ndarray


class CustomTrainState(TrainState):
    # influence weights over the dynamic coordinates (zero outside S^d, zero before the partition)
    w_ex: jnp.ndarray
    # automatic partition: boolean mask of S^d, its readiness flag and the last action-sensitivity scores
    D: jnp.ndarray
    part_ready: jnp.ndarray
    chi: jnp.ndarray
    # running statistics of the world state / one-step changes (frozen after STATS_FRAC of training)
    s_mean: jnp.ndarray
    s_m2: jnp.ndarray
    s_cnt: jnp.ndarray
    ds_m2: jnp.ndarray
    ds_cnt: jnp.ndarray
    timesteps: int = 0
    n_updates: int = 0


def two_means_split(score, log_scale=True):
    """Exact 1-D two-means split of the normalised scores. Returns a boolean mask of the high cluster.

    Scores span orders of magnitude (fully vs. partially action-driven coordinates), so the split is
    computed on log(score + 0.01) by default."""
    v = score / (jnp.max(score) + 1e-8)
    if log_scale:
        v = jnp.log(v + 0.01)
    d = v.shape[0]
    s = jnp.sort(v)
    c1 = jnp.cumsum(s)
    c2 = jnp.cumsum(s ** 2)
    k = jnp.arange(1, d)  # size of the low cluster
    tot1, tot2 = c1[-1], c2[-1]
    low_sse = c2[k - 1] - c1[k - 1] ** 2 / k
    high_n = d - k
    high_sse = (tot2 - c2[k - 1]) - (tot1 - c1[k - 1]) ** 2 / high_n
    best = jnp.argmin(low_sse + high_sse)
    thresh = s[best + 1]  # smallest value of the high cluster
    return v >= thresh


def sample_same_episode(key, global_done, k):
    """For every anchor step t and batch element, sample k transition indices from the same episode.

    global_done [T, B]: the episode ended after step t. Transition j (state_j -> state_{j+1}) is
    valid if ~global_done[j]; only the first T-1 transitions of the window have a next state.
    Returns idx [T-1, B, k] and a validity mask [T-1, B, k]."""
    T, B = global_done.shape
    Tm = T - 1
    ep = jnp.cumsum(jnp.concatenate([jnp.zeros((1, B)), global_done[:-1].astype(jnp.float32)], 0), 0)[:Tm]
    valid = ~global_done[:Tm]
    same = (ep[:, None, :] == ep[None, :, :]) & valid[None, :, :]  # [Tm (anchor), Tm (sample), B]
    scores = jnp.where(same, jax.random.uniform(key, same.shape), -1.0)
    top_v, top_i = jax.lax.top_k(jnp.moveaxis(scores, 1, -1), k)  # [Tm, B, k]
    return top_i, (top_v >= 0).astype(jnp.float32)


def make_train(config):
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"] + 1
    training_set = list(config["TRAINING_SET"])
    test_set = list(config["TEST_SET"]) + training_set

    env_kwargs = dict(config["ENV_KWARGS"])
    env_kwargs["scenario"] = map_name_to_scenario(config["MAP_NAME"])

    env = HeuristicEnemySMAX(task_ids=training_set, **env_kwargs)
    env = SMAXWorldStateWrapper(env)
    env = SMAXLogWrapper(env)
    wrapped_env = SMAXRolloutManager(env, batch_size=config["NUM_ENVS"], ally_agents=env.ally_agents)

    test_env = HeuristicEnemySMAX(task_ids=test_set, **env_kwargs)
    test_env = SMAXLogWrapper(test_env)
    test_env = SMAXRolloutManager(test_env, batch_size=config["TEST_NUM_ENVS"], ally_agents=test_env.ally_agents)

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.ally_agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.ally_agents)}

    N = config["NUM_AGENTS"] = env.num_allies
    E = config["NUM_ENVS"]
    T = config["NUM_STEPS"]
    S = config["STATE_DIM"] = wrapped_env.world_state_size()
    A = config["ACT_DIM"] = wrapped_env.max_action_space
    config["OBS_DIM"] = wrapped_env.obs_size
    NUM_UPDATES = config["NUM_UPDATES"]
    n_part = max(1, int(NUM_UPDATES * config["PARTITION_WARMUP_FRAC"]))
    n_stats = max(1, int(NUM_UPDATES * config["STATS_FRAC"]))

    # ---- world-state coordinates ------------------------------------------------------------
    # world state = per unit [health, x, y, cooldown, unit-type one-hot], then the agent one-hot.
    # The dynamic coordinates (health, x, y, cooldown of every unit) are the partition candidates;
    # unit-type bits and the agent one-hot are static.
    S0 = env.state_size
    F = len(env.own_features)
    U = env.num_agents  # allies + enemies
    dyn_idx_np = np.asarray([u * F + f for u in range(U) for f in range(4)])
    d = config["DYN_DIM"] = int(dyn_idx_np.shape[0])
    dyn_idx = jnp.asarray(dyn_idx_np)
    alive_idx = jnp.asarray((dyn_idx_np // F) * F)  # health coordinate of the owning unit
    valid_on_death = jnp.asarray((dyn_idx_np % F) == 0)  # a health change at death is meaningful
    coord_names = [f"{'ally' if u < N else 'enemy'}{u if u < N else u - N}_{nm}"
                   for u in range(U) for nm in ("hp", "x", "y", "cd")]
    # domain-informed partition of the paper (LEICA (manual)): S^d = ally positions, ally cooldowns, enemy health
    manual_D = jnp.asarray([(u < N and f > 0) or (u >= N and f == 0) for u in range(U) for f in range(4)])
    auto_partition = config.get("PARTITION", "auto") == "auto"

    def coord_mask(state, next_state, done):
        """Per-coordinate validity of a transition [..., d]: in-episode, the owning unit alive at t
        and (unless the coordinate is meaningful at death) also alive at t+1, because the
        features of a dead unit are zeroed."""
        alive_t = state[..., alive_idx] > 0
        alive_n = next_state[..., alive_idx] > 0
        return (~done)[..., None] & alive_t & (valid_on_death | alive_n)

    actor = ActorDRNN(A, config=config)
    critic = CriticDRNN(config=config)
    encoder = VAEncoder(config=config)
    decoder = InfluenceDecoder(config=config, out_dim=d)
    f_part = MLP(d)  # joint one-step model f(S, A) -> standardised change of the dynamic coordinates
    g_part = MLP(d)  # action-free one-step model g(S) (used by the "gain" partition score)

    def train(rng):

        def initialize_params(rng):
            rng, rng_a, rng_c, rng_d, rng_e, rng_f, rng_g = jax.random.split(rng, 7)

            actor_init = (jnp.zeros((1, 1, 2 * config["Z_DIM"])), jnp.zeros((1, 1, config["OBS_DIM"])), jnp.zeros((1, 1)), jnp.zeros((1, 1, A)))
            init_hs_actor = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)
            actor_params = actor.init(rng_a, init_hs_actor, *actor_init)

            init_state = (jnp.zeros((1, 1, 2 * config["Z_DIM"])), jnp.zeros((1, 1, S)), jnp.zeros((1, 1)))
            init_hs_critic = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)
            critic_params = critic.init(rng_c, init_hs_critic, *init_state)

            init_hs_encoder = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)
            encoder_params = encoder.init(rng_e, init_hs_encoder, jnp.zeros((1, 1, config["OBS_DIM"])), jnp.zeros((1, 1)))

            decoder_params = decoder.init(rng_d, jnp.zeros((1, 1, config["Z_DIM"])), jnp.zeros((1, 1, config["Z_DIM"])), jnp.zeros((1, 1, S)), jnp.zeros((1, 1, A)))
            f_params = f_part.init(rng_f, jnp.zeros((1, S0 + N * A)))
            g_params = g_part.init(rng_g, jnp.zeros((1, S0)))

            return {"actor": actor_params, "critic": critic_params, "decoder": decoder_params, "encoder": encoder_params,
                    "fpart": f_params, "gpart": g_params}

        def create_agent(rng):
            params = initialize_params(rng)

            lr_scheduler = optax.linear_schedule(init_value=config["LR"], end_value=1e-10, transition_steps=config["NUM_EPOCHS"] * NUM_UPDATES)
            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=lr))

            train_state = CustomTrainState.create(
                apply_fn=actor.apply, params=params, tx=tx,
                w_ex=jnp.zeros((d,)) if auto_partition else manual_D / manual_D.sum(),
                D=jnp.ones((d,), dtype=bool) if auto_partition else manual_D,
                part_ready=jnp.asarray(not auto_partition), chi=jnp.zeros((d,)),
                s_mean=jnp.zeros((S0,)), s_m2=jnp.ones((S0,)), s_cnt=jnp.zeros(()),
                ds_m2=jnp.ones((d,)), ds_cnt=jnp.zeros((d,)),
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(_rng)

        init_hs_encoder = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], N, E)
        init_hs_actor = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], N, E)
        init_hs_critic = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], N, E)

        def stats_of(ts):
            s_std = jnp.sqrt(ts.s_m2 / jnp.maximum(ts.s_cnt, 1)) + 1e-3
            ds_std = jnp.sqrt(ts.ds_m2 / jnp.maximum(ts.ds_cnt, 1)) + 1e-3
            return ts.s_mean, s_std, ds_std

        def part_inputs(ts, s_cur, s_next, actions):
            """Standardised inputs/targets of the one-step models. s_cur, s_next [.., S0]; actions [.., N]."""
            s_mean, s_std, ds_std = stats_of(ts)
            s_std_in = (s_cur - s_mean) / s_std
            dS = (s_next[..., dyn_idx] - s_cur[..., dyn_idx]) / ds_std
            joint = jax.nn.one_hot(actions, A).reshape(*actions.shape[:-1], N * A)
            return s_std_in, dS, joint

        def estimate_partition(ts, s_cur, s_next, actions, avail, done, rng):
            """Action-sensitivity scores of the dynamic coordinates and the two-means split.

            cf_j   = E |f_j(S, A^{i<-a'}) - f_j(S, A)| / sigma_j  (counterfactual single-agent action
                     perturbations over valid alternative actions, averaged over agents and samples;
                     sigma_j = standard deviation of coordinate j), the score of the paper;
            gain_j = 1 - MSE_j(f(S, A)) / MSE_j(g(S)): fraction of the one-step change explained by
                     the team action (alternative, PARTITION_SCORE="gain")."""
            fp, gp = ts.params["fpart"], ts.params["gpart"]
            _, s_std, ds_std = stats_of(ts)
            s_in, dS, joint = part_inputs(ts, s_cur, s_next, actions)
            cm = coord_mask(s_cur, s_next, done).astype(jnp.float32)
            ef = ((f_part.apply(fp, jnp.concatenate([s_in, joint], -1)) - dS) ** 2 * cm).sum((0, 1))
            eg = ((g_part.apply(gp, s_in) - dS) ** 2 * cm).sum((0, 1))
            gain = jnp.clip(1.0 - ef / jnp.maximum(eg, 1e-8), 0.0, 1.0)

            Tm = s_in.shape[0]
            M = config["CHI_SAMPLES"]
            pick = jax.random.randint(rng, (M,), 0, Tm * E)
            t_i, e_i = pick // E, pick % E
            st, act, av = s_in[t_i, e_i], actions[t_i, e_i], avail[t_i, e_i]

            def f_of(s_, acts):
                return f_part.apply(fp, jnp.concatenate([s_, jax.nn.one_hot(acts, A).reshape(-1)], -1))

            def per_sample(s_, acts, avm):
                base = f_of(s_, acts)

                def per_agent(i):
                    diffs = jax.vmap(lambda a: jnp.abs(f_of(s_, acts.at[i].set(a)) - base))(jnp.arange(A))
                    m = avm[i] * (jnp.arange(A) != acts[i])
                    return (diffs * m[:, None]).sum(0), m.sum()

                sums, cnts = jax.vmap(per_agent)(jnp.arange(N))
                return sums.sum(0), cnts.sum()

            sums, cnts = jax.vmap(per_sample)(st, act, av)
            cf = sums.sum(0) / jnp.maximum(cnts.sum(), 1)  # in units of the one-step-change std
            cf = cf * ds_std / s_std[dyn_idx]  # -> normalised by the coordinate std (paper)
            score = cf if config["PARTITION_SCORE"] == "cf" else gain
            return score, two_means_split(score)

        def print_partition(at, n, D, score):
            if not bool(np.asarray(at).reshape(-1)[0]):
                return
            names = [nm for nm, m in zip(coord_names, np.asarray(D).reshape(-1)) if m]
            sc = np.asarray(score).reshape(-1)
            print(f"[update {int(np.asarray(n).reshape(-1)[0])}] automatic partition: {len(names)}/{d} coordinates in S^d: "
                  + ", ".join(names))
            print("  scores: " + ", ".join(f"{nm}={s:.3g}" for nm, s in zip(coord_names, sc)), flush=True)

        def _update_step(runner_state, unused):
            train_state, env_state, test_state, last_obs, last_rewards, last_dones, hstates, rng = runner_state

            def _step_env(carry, _):
                env_state, last_obs, last_rewards, last_dones, hstates, w_ex, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                obs, dones, rewards = batchify(last_obs), batchify(last_dones), batchify(last_rewards)

                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
                avail = jax.lax.stop_gradient(batchify(avail_actions))

                hs_encoder, z_int, _, z_ext_mu, z_ext_std = jax.vmap(encoder.apply, in_axes=(None, 0, 0, 0))(train_state.params["encoder"], hstates[0], obs[:, np.newaxis], dones[:, np.newaxis])
                z = jnp.concatenate([z_ext_mu, z_ext_std], axis=-1)

                hs_actor, pi = jax.vmap(actor.apply, in_axes=(None, 0, 0, 0, 0, 0))(train_state.params["actor"], hstates[1], jax.lax.stop_gradient(z), obs[:, np.newaxis], dones[:, np.newaxis], avail)

                actions = pi.sample(seed=rng_a)
                log_prob = pi.log_prob(actions).squeeze(axis=1)
                actions = actions.squeeze(axis=1)

                ally_state = last_obs["world_state"].swapaxes(0, 1)

                hs_critic, value = jax.vmap(critic.apply, in_axes=(None, 0, 0, 0, 0))(train_state.params["critic"], hstates[2], jax.lax.stop_gradient(z), ally_state[:, jnp.newaxis], dones[:, jnp.newaxis])
                value = value.squeeze(axis=1)
                new_obs, new_env_state, rewards, _dones, infos = wrapped_env.batch_step(rng_s, env_state, unbatchify(actions))
                _rewards = config.get("REW_SCALE", 1) * batchify(rewards)
                global_done = jnp.tile(_dones["__all__"][jnp.newaxis, :], (N, 1))

                batch = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), (z_int.squeeze(1), z_ext_mu.squeeze(1), ally_state))

                def decode_all(z_int, z_ext, s):
                    """Predicted next S^d under every action of this agent -> [A, d]."""
                    tile = lambda x: jnp.broadcast_to(x, (A,) + x.shape)
                    pred_int, _ = decoder.apply(train_state.params["decoder"], tile(z_int), tile(z_ext), tile(s), jnp.eye(A), training=False)
                    return pred_int

                # influence reward (Eq. cf): weighted magnitude of the direct change under the taken action,
                # relative to its expectation under the policy over valid actions (w_ex is zero outside S^d)
                preds = jax.vmap(decode_all)(*batch)                                        # [B, A, d]
                gaps = jnp.abs(preds - batch[-1][:, None, dyn_idx])                         # [B, A, d]
                Sgap = jnp.take_along_axis(gaps, actions.reshape(-1)[:, None, None], 1)[:, 0]
                p_valid = pi.probs.squeeze(1).reshape((-1, A)) * avail.reshape((-1, A))
                p_valid = p_valid / jnp.maximum(p_valid.sum(-1, keepdims=True), 1e-8)
                Sgap_bar = jnp.einsum("ba,bad->bd", p_valid, gaps)
                Sgapdiff = jax.lax.stop_gradient(Sgap - Sgap_bar).reshape(N, E, -1) * w_ex
                Ri = Sgapdiff.sum(-1)
                Ri_mean = jnp.broadcast_to(Ri.mean(axis=0, keepdims=True), Ri.shape)

                transition = Transition(
                    global_done=global_done,
                    obs=obs,
                    state=ally_state,
                    actions=actions,
                    rewards=_rewards,
                    log_probs=log_prob,
                    values=value,
                    dones=dones,
                    avail_actions=avail,
                    r_indiv=Ri,
                    r_team=Ri_mean,
                )
                runner_state = (new_env_state, new_obs, rewards, _dones, (hs_encoder, hs_actor, hs_critic), w_ex, rng)
                return runner_state, (transition, infos)

            init_hstates = runner_state[-2]

            expl_state = (env_state, last_obs, last_rewards, last_dones, hstates, train_state.w_ex[None, None])
            rng, _rng = jax.random.split(rng)
            expl_state, (traj_batch, infos) = jax.lax.scan(_step_env, (*expl_state, _rng), None, T)
            env_state, last_obs, last_rewards, last_dones, hstates, w_ex, rng = expl_state

            # per-environment view of the rollout for the one-step models of the automatic partition
            # (world state without the agent one-hot, joint actions): T-1 transitions with a next state
            part_s = traj_batch.state[:, 0, :, :S0]                       # [T, E, S0]
            part_a = jnp.moveaxis(traj_batch.actions, 1, 2)               # [T, E, N]
            part_av = jnp.moveaxis(traj_batch.avail_actions, 1, 2)        # [T, E, N, A]
            part_done = traj_batch.global_done[:, 0]                      # [T, E]
            part_batch = (part_s[:-1], part_s[1:], part_a[:-1], part_av[:-1], part_done[:-1])

            # running statistics (frozen after n_stats updates)
            n = train_state.n_updates
            s_cur, s_next, _, _, p_done = part_batch
            valid = (~p_done).astype(jnp.float32).reshape(-1, 1)
            s_flat = s_cur.reshape(-1, S0)
            ds = (s_next[..., dyn_idx] - s_cur[..., dyn_idx]).reshape(-1, d)
            cm = coord_mask(s_cur, s_next, p_done).reshape(-1, d).astype(jnp.float32)
            bc = valid.sum()
            bmean = (s_flat * valid).sum(0) / jnp.maximum(bc, 1)
            bm2 = (((s_flat - bmean) ** 2) * valid).sum(0)
            new_cnt = train_state.s_cnt + bc
            delta = bmean - train_state.s_mean
            mean = train_state.s_mean + delta * bc / jnp.maximum(new_cnt, 1)
            m2 = train_state.s_m2 + bm2 + delta ** 2 * train_state.s_cnt * bc / jnp.maximum(new_cnt, 1)
            upd = n < n_stats
            train_state = train_state.replace(
                s_mean=jnp.where(upd, mean, train_state.s_mean),
                s_m2=jnp.where(upd, m2, train_state.s_m2),
                s_cnt=jnp.where(upd, new_cnt, train_state.s_cnt),
                ds_m2=jnp.where(upd, train_state.ds_m2 + ((ds ** 2) * cm).sum(0), train_state.ds_m2),
                ds_cnt=jnp.where(upd, train_state.ds_cnt + cm.sum(0), train_state.ds_cnt),
            )

            rng, rng_z = jax.random.split(rng)

            obs, dones, rewards = batchify(last_obs), batchify(last_dones), batchify(last_rewards)

            hs_encoder, z_int_mu, _, z_ext_mu, z_ext_std = jax.vmap(encoder.apply, in_axes=(None, 0, 0, 0))(train_state.params["encoder"], hstates[0], obs[:, np.newaxis], dones[:, np.newaxis])
            z = jnp.concatenate([z_ext_mu, z_ext_std], axis=-1)

            ally_state = last_obs["world_state"].swapaxes(0, 1)

            _, last_val = jax.vmap(critic.apply, in_axes=(None, 0, 0, 0, 0))(train_state.params["critic"], hstates[2], jax.lax.stop_gradient(z), ally_state[:, jnp.newaxis], dones[:, jnp.newaxis])
            last_val = last_val.squeeze(axis=1).reshape(-1)

            traj_batch = jax.tree.map(lambda x: x.reshape(x.shape[0], -1, *x.shape[3:]), traj_batch)

            def _calculate_gae(traj_batch, last_values):
                """GAE on the shaped rewards: r_ext + lambda r_int^i for the actor of agent i, and
                r_ext + lambda mean_i r_int^i for the critic target."""
                def _get_advantages(carry, transition):
                    gae_i, gae_t, next_value = carry
                    done, value, reward, r_i, r_t = transition.global_done, transition.values, transition.rewards, transition.r_indiv, transition.r_team
                    delta_i = reward + config["REWARD_WEIGHT"] * r_i + config["GAMMA"] * next_value * (1 - done) - value
                    gae_i = delta_i + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae_i
                    delta_t = reward + config["REWARD_WEIGHT"] * r_t + config["GAMMA"] * next_value * (1 - done) - value
                    gae_t = delta_t + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae_t

                    return (gae_i, gae_t, value), (gae_i, gae_t + value)

                zeros = jnp.zeros_like(last_values)
                _, (advantages, target_values) = jax.lax.scan(_get_advantages, (zeros, zeros, last_values), traj_batch, reverse=True, unroll=16)
                return advantages, target_values

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info, rng):
                    init_hstates, traj_batch, advantages, targets = batch_info
                    init_hstate_encoder, init_hstate_actor, init_hstate_critic = init_hstates
                    Dm = train_state.D.astype(jnp.float32)
                    Rm = jnp.where(train_state.part_ready, 1.0 - Dm, 1.0)  # before the partition both heads see all coordinates

                    def _loss_fn(params, gae, rng):
                        _, z_int_mu, z_int_std, z_ext_mu, z_ext_std = encoder.apply(params["encoder"], init_hstate_encoder, traj_batch.obs, traj_batch.dones)
                        rng, rng_z = jax.random.split(rng)
                        z_ext = distrax.MultivariateNormalDiag(z_ext_mu, z_ext_std).sample(seed=rng_z)
                        rng, rng_z = jax.random.split(rng)
                        z_int = distrax.MultivariateNormalDiag(z_int_mu, z_int_std).sample(seed=rng_z)
                        z = jnp.concatenate([z_ext_mu, z_ext_std], axis=-1)
                        _, pi = actor.apply(params["actor"], init_hstate_actor, jax.lax.stop_gradient(z), traj_batch.obs, traj_batch.dones, traj_batch.avail_actions)

                        log_prob = pi.log_prob(traj_batch.actions)

                        one_hot_actions = jax.nn.one_hot(traj_batch.actions, A)
                        B = traj_batch.obs.shape[1]

                        # one-step predictors: K transitions sampled from the same episode as the context
                        rng, _rng, rng_k = jax.random.split(rng, 3)
                        dropout_rngs = jax.random.split(_rng, config["Z_SAMPLES"])
                        idx, traj_mask = sample_same_episode(rng_k, traj_batch.global_done, config["Z_SAMPLES"])
                        b_idx = jnp.arange(B)[None, :, None]
                        batch = (traj_batch.state[:-1], one_hot_actions[:-1], traj_batch.state[1:, :, dyn_idx])
                        s, ja, target = jax.tree.map(lambda x: x[idx, b_idx], batch)  # [T-1, B, K, ...]
                        traj_mask = traj_mask[..., None]

                        pred_int, pred_ext = jax.vmap(decoder.apply, in_axes=(None, None, None, 2, 2), out_axes=(2, 2))(params["decoder"], z_int[:-1], z_ext[:-1], s, ja, rngs={'dropout': dropout_rngs})
                        sq_int = (jnp.square(pred_int - target) * traj_mask).mean(2) / (traj_mask.mean(2) + 1e-10)
                        sq_ext = (jnp.square(pred_ext - target) * traj_mask).mean(2) / (traj_mask.mean(2) + 1e-10)
                        int_loss = config["A_RECON"] * 0.5 * (sq_int * Dm).sum(-1) / jnp.maximum(Dm.sum(), 1)
                        ext_loss = config["A_RECON"] * 0.5 * (sq_ext * Rm).sum(-1) / jnp.maximum(Rm.sum(), 1)
                        int_loss = int_loss.mean(0).reshape(N, E).mean(0)
                        ext_loss = ext_loss.mean(0).reshape(N, E).mean(0)

                        def elbo(mu, std, r_loss):
                            std = jnp.exp(jnp.log(std) * ~traj_batch.global_done[..., None])
                            mu = mu * ~traj_batch.global_done[..., None]

                            q = distrax.MultivariateNormalDiag(mu[1:], std[1:])
                            prior = distrax.MultivariateNormalDiag(mu[:-1], std[:-1])
                            kl_loss = config["A_KL"] * q.kl_divergence(prior)
                            kl_loss = kl_loss.mean(0).reshape(N, E).mean(0)

                            return (r_loss + kl_loss).mean()

                        elbo_loss = elbo(z_ext_mu, z_ext_std, ext_loss) + elbo(z_int_mu, z_int_std, int_loss)

                        # auxiliary one-step models of the automatic partition
                        p_s, p_sn, p_a, _, p_d = part_batch
                        s_in, dS, joint = part_inputs(train_state, p_s, p_sn, p_a)
                        pcm = coord_mask(p_s, p_sn, p_d).astype(jnp.float32)
                        pred_f = f_part.apply(params["fpart"], jnp.concatenate([s_in, joint], -1))
                        pred_g = g_part.apply(params["gpart"], s_in)
                        f_loss = (((pred_f - dS) ** 2) * pcm).sum() / jnp.maximum(pcm.sum(), 1)
                        g_loss = (((pred_g - dS) ** 2) * pcm).sum() / jnp.maximum(pcm.sum(), 1)

                        logratio = log_prob - traj_batch.log_probs
                        ratio = jnp.exp(logratio)

                        gae_normalized = (gae - gae.mean()) / (gae.std() + 1e-8)

                        loss_actor1 = ratio * gae_normalized
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_normalized
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        actor_loss = loss_actor - config["ENT_COEF"] * entropy

                        _, value = critic.apply(params["critic"], init_hstate_critic, jax.lax.stop_gradient(z), traj_batch.state, traj_batch.dones)

                        value_pred_clipped = traj_batch.values + (value - traj_batch.values).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped)).mean()
                        critic_loss = config["VF_COEF"] * value_loss

                        total_loss = actor_loss + elbo_loss + critic_loss + f_loss + g_loss

                        logs = {
                            "total_loss": total_loss,
                            "ppo_loss": loss_actor,
                            "entropy": entropy,
                            "approx_kl": approx_kl,
                            "clip_frac": clip_frac,
                            "int_loss": int_loss.mean(),
                            "ext_loss": ext_loss.mean(),
                            "f_loss": f_loss,
                            "g_loss": g_loss,
                            "critic_loss": critic_loss,
                            "elbo_loss": elbo_loss,
                            "values": traj_batch.values.mean(),
                            "rewards": traj_batch.rewards.mean(),
                            "ri_mean": traj_batch.r_indiv.mean(),
                            "ri_std": traj_batch.r_indiv.std(),
                            "advantages": gae.mean(),
                        }

                        return total_loss, (logs, rng)

                    loss, grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params, advantages, rng)
                    (logs, rng) = loss[1]
                    train_state = train_state.apply_gradients(grads=grads)
                    loss_info = {k: v.mean() for k, v in logs.items()}

                    return train_state, loss_info, rng

                (train_state, init_hstates, traj_batch, advantages, targets, rng) = update_state
                batch = (init_hstates, traj_batch, advantages, targets)
                train_state, loss_info, rng = _update_minibatch(train_state, batch, rng)
                update_state = (train_state, init_hstates, traj_batch, advantages, targets, rng)

                return update_state, loss_info

            def _update_w(update_state):
                """Influence weights over S^d from the response-predictor Jacobian (mediated influence,
                kappa) and the critic gradient (value sensitivity, 1 - kappa): batch average, standardisation
                across coordinates, temperature softmax and EMA. Only applied once the partition is available."""
                (train_state, init_hstates, traj_batch, advantages, targets, rng) = update_state
                _, z_int_mu, z_int_std, z_ext_mu, z_ext_std = encoder.apply(train_state.params["encoder"], init_hstates[0], traj_batch.obs, traj_batch.dones)
                z = jnp.concatenate([z_ext_mu, z_ext_std], axis=-1)
                Dm = train_state.D.astype(jnp.float32)
                Rm = 1.0 - Dm

                def get_jac(z_int, z_ext, s, joint_actions):
                    def f(s):
                        _, pred_ext = decoder.apply(train_state.params["decoder"], z_int, z_ext, s, joint_actions, training=False)
                        return pred_ext
                    return jax.jacrev(f)(s)  # [d_out, S]

                def make_g(state):
                    _, g = critic.apply(train_state.params["critic"], init_hstates[2], jax.lax.stop_gradient(z), state, traj_batch.dones)
                    return g.sum()

                kappa = config["ALPHA"]
                if config["ALPHA_OPTION"] == "exp":
                    tau = 10
                    kappa = jnp.exp(-tau * train_state.n_updates / NUM_UPDATES)
                elif config["ALPHA_OPTION"] == "linear":
                    kappa = 1 - train_state.n_updates / NUM_UPDATES

                one_hot_actions = jax.nn.one_hot(traj_batch.actions, A)

                # Jacobian of the response prediction p_y(S_{t+1}, y_t) w.r.t. S^d_{t+1} (Eq. jacobian): next states
                # of the batch paired with the contexts inferred at t
                batch_jac = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]),
                                         (z_int_mu[:-1], z_ext_mu[:-1], traj_batch.state[1:], one_hot_actions[:-1]))

                J_x = jax.vmap(get_jac)(*batch_jac)[..., dyn_idx] * Rm[None, :, None]  # response outputs x dynamic inputs
                J_x = jnp.sqrt(jnp.sum(J_x ** 2, axis=1)).reshape(T - 1, N * E, -1)    # column norms over the S^r outputs

                # value sensitivity |dV(S_{t+1}) / dS^d_{t+1}| (Eq. cv)
                J_v = jax.grad(make_g)(traj_batch.state)[1:]
                J_v = jnp.abs(J_v[..., dyn_idx])

                def zstd(c):  # standardise the batch-averaged scores across the S^d coordinates
                    m = (c * Dm).sum() / jnp.maximum(Dm.sum(), 1)
                    s = jnp.sqrt((((c - m) ** 2) * Dm).sum() / jnp.maximum(Dm.sum(), 1))
                    return (c - m) / (s + 1e-8)

                c = kappa * zstd(J_x.mean(axis=(0, 1))) + (1 - kappa) * zstd(J_v.mean(axis=(0, 1)))

                w_new = jax.lax.stop_gradient(jax.nn.softmax(jnp.where(Dm > 0, c / config["W_TAU"], -1e9)))
                w_ema = (1 - config["EMA_RATE"]) * train_state.w_ex + config["EMA_RATE"] * w_new
                train_state = train_state.replace(w_ex=jnp.where(train_state.part_ready, w_ema, train_state.w_ex))

                update_state = (train_state, init_hstates, traj_batch, advantages, targets, rng)
                return update_state

            init_hstates = jax.tree.map(lambda x: x.reshape(-1, config["HIDDEN_SIZE"]), init_hstates)
            update_state = (train_state, init_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_infos = jax.lax.scan(_update_epoch, update_state, None, config["NUM_EPOCHS"])
            train_state = update_state[0]

            # automatic partition: estimated once, at the end of the warm-up window, then fixed
            rng, rng_p = jax.random.split(update_state[-1])
            at = (train_state.n_updates == n_part - 1) & auto_partition
            score, D_new = jax.lax.cond(
                at,
                lambda: estimate_partition(train_state, *part_batch, rng_p),
                lambda: (jnp.zeros((d,)), jnp.ones((d,), dtype=bool)),
            )
            train_state = train_state.replace(
                D=jnp.where(at, D_new, train_state.D),
                chi=jnp.where(at, score, train_state.chi),
                part_ready=train_state.part_ready | at,
                w_ex=jnp.where(at, D_new / jnp.maximum(D_new.sum(), 1), train_state.w_ex),
            )
            jax.debug.callback(print_partition, at, train_state.n_updates, train_state.D, train_state.chi)
            update_state = (train_state, *update_state[1:-1], rng)
            update_state = _update_w(update_state)
            train_state = update_state[0]

            infos = jax.tree.map(lambda x: jnp.nanmean(jnp.where(infos["returned_episode"], x, jnp.nan), axis=(0, 1)), infos)

            metrics = jax.tree.map(lambda x: x.mean(), loss_infos)
            metrics = metrics | {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "w_ex_std": train_state.w_ex.std(),
                "n_direct": train_state.D.sum(),
            }
            metrics.update({"train/" + k: v[0] for k, v in infos.items()})

            test_interval = max(1, int(NUM_UPDATES * config["TEST_INTERVAL"]))
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates % test_interval == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics["test/seen/won"] = test_state["seen"]["returned_won_episode"]
                metrics["test/unseen/won"] = test_state["unseen"]["returned_won_episode"]

                def callback(n, env_step, train_won, seen, unseen):
                    n = int(np.asarray(n).reshape(-1)[0])
                    if n % test_interval == 0:
                        print(f"update {n:5d} | env steps {int(np.asarray(env_step).reshape(-1)[0]):>10d} | "
                              f"train win {float(np.nanmean(train_won)):.3f} | test win seen {float(np.nanmean(seen)):.3f} "
                              f"unseen {float(np.nanmean(unseen)):.3f}", flush=True)
                jax.debug.callback(callback, train_state.n_updates, train_state.timesteps, metrics["train/returned_won_episode"],
                                   metrics["test/seen/won"], metrics["test/unseen/won"])

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            train_state = train_state.replace(timesteps=train_state.timesteps + T * E)
            runner_state = (train_state, env_state, test_state, last_obs, last_rewards, last_dones, hstates, rng)
            return runner_state, metrics

        def get_greedy_metrics(rng, train_state):
            """Greedy evaluation on all training and test regimes (TEST_NUM_ENVS episodes in parallel)."""
            actor_params = train_state.params["actor"]
            encoder_params = train_state.params["encoder"]

            def _greedy_env_step(step_state, unused):
                encoder_params, actor_params, env_state, last_obs, last_rewards, last_dones, hs_encoder, hs, rng = step_state
                rng, _rng = jax.random.split(rng)

                obs = batchify(last_obs)[:, np.newaxis]
                dones = batchify(last_dones)[:, np.newaxis]
                avail_actions = test_env.get_valid_actions(env_state.env_state)
                avail = batchify(avail_actions)

                hs_encoder, _, _, z_ext_mu, z_ext_std = jax.vmap(encoder.apply, in_axes=(None, 0, 0, 0))(encoder_params, hs_encoder, obs, dones)
                z = jnp.concatenate([z_ext_mu, z_ext_std], axis=-1)
                label = env_state.env_state.enemy_policy_state.enemy_policy_index[..., 0]

                hs, pi = jax.vmap(actor.apply, in_axes=(None, 0, 0, 0, 0, 0))(actor_params, hs, z, obs, dones, avail)
                actions = pi.mode().squeeze(axis=1)
                actions = unbatchify(actions)

                obs, env_state, rewards, dones, test_infos = test_env.batch_step(_rng, env_state, actions)

                step_state = (encoder_params, actor_params, env_state, obs, rewards, dones, hs_encoder, hs, rng)
                return step_state, (label, test_infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool) for agent in env.ally_agents + ["__all__", "ally", "enemy"]}
            init_rewards = {agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=float) for agent in env.ally_agents + ["__all__"]}
            rng, _rng = jax.random.split(rng)
            hs_encoder = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], len(env.ally_agents), config["TEST_NUM_ENVS"])
            hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], len(env.ally_agents), config["TEST_NUM_ENVS"])

            step_state = (encoder_params, actor_params, env_state, init_obs, init_rewards, init_dones, hs_encoder, hs, _rng)

            step_state, (labels, test_infos) = jax.lax.scan(_greedy_env_step, step_state, None, config["TEST_NUM_STEPS"])
            test_infos = jax.tree.map(lambda x: jnp.nanmean(jnp.where(test_infos["returned_episode"], x, jnp.nan), axis=2), test_infos)
            test_info_indiv = {}
            seen = test_infos["returned_episode"].astype(bool) & (labels >= len(config["TEST_SET"]))
            unseen = test_infos["returned_episode"].astype(bool) & (labels < len(config["TEST_SET"]))
            test_info_indiv["seen"] = jax.tree.map(lambda x: jnp.nanmean(jnp.where(seen, x, jnp.nan), axis=(0, 1)), test_infos)
            test_info_indiv["unseen"] = jax.tree.map(lambda x: jnp.nanmean(jnp.where(unseen, x, jnp.nan), axis=(0, 1)), test_infos)
            for task in range(len(test_set)):
                test_info_indiv[f'{task}'] = jax.tree.map(lambda x: jnp.nanmean(jnp.where(test_infos["returned_episode"].astype(bool) & (labels == task), x, jnp.nan), axis=(0, 1)), test_infos)

            return test_info_indiv

        rng, _rng = jax.random.split(rng)
        last_obs, env_state = wrapped_env.batch_reset(_rng)
        last_done = {agent: jnp.zeros((E), dtype=bool) for agent in env.ally_agents + ["__all__", "ally", "enemy"]}
        last_reward = {agent: jnp.zeros((E), dtype=float) for agent in env.ally_agents + ["__all__"]}

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)
        rng, _rng = jax.random.split(rng)
        hstates = (init_hs_encoder, init_hs_actor, init_hs_critic)
        runner_state = (train_state, env_state, test_state, last_obs, last_reward, last_done, hstates, rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, NUM_UPDATES)

        return {"runner_state": runner_state, "metrics": metrics}

    train.coord_names = coord_names
    train.n_part = n_part
    return train
