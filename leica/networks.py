"""Networks used by LEICA (recurrent actor/critic, context encoder, one-step predictors)."""
from functools import partial
from typing import Dict, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(hidden_size, *ins.shape[:-1]), rnn_state)
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(jax.random.PRNGKey(0), (*batch_size, hidden_size))


class ActorDRNN(nn.Module):
    """Recurrent actor conditioned on the local observation history and the regime context."""
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, z, obs, dones, avail_actions):
        act = lambda x: nn.tanh(x)

        obs_embed = nn.Dense(self.config["HIDDEN_SIZE"])(obs)
        obs_embed = act(obs_embed)
        rnn_in = (obs_embed, dones)
        hidden, obs_embed = ScannedRNN()(hidden, rnn_in)
        actor_mean = nn.Dense(self.config["HIDDEN_SIZE"])(jnp.concatenate([obs_embed, z], axis=-1))

        actor_mean = act(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticDRNN(nn.Module):
    """Centralised recurrent critic on the world state and the regime context."""
    config: Dict

    @nn.compact
    def __call__(self, hidden, z, world_state, dones):
        norm = lambda x: nn.LayerNorm()(x)
        act = lambda x: nn.tanh(x)

        embedding = act(nn.Dense(self.config["HIDDEN_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(world_state))

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = act(nn.Dense(self.config["HIDDEN_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(jnp.concatenate([embedding, z], axis=-1)))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(norm(critic))

        return hidden, jnp.squeeze(critic, axis=-1)


class VAEncoder(nn.Module):
    """History encoder (shared GRU) with two Gaussian heads: the direct context z and the regime
    context y (named z_int / z_ext in the code)."""
    config: Dict

    @nn.compact
    def __call__(self, hidden, obs, dones):
        act = lambda x: nn.tanh(x)

        obs_embed = act(nn.Dense(self.config["HIDDEN_SIZE"])(obs))
        rnn_in = (obs_embed, dones)
        hidden, obs_embed = ScannedRNN()(hidden, rnn_in)

        int_embed = act(nn.Dense(self.config["HIDDEN_SIZE"])(obs_embed))
        ext_embed = act(nn.Dense(self.config["HIDDEN_SIZE"])(obs_embed))

        z_int_mu = nn.Dense(self.config["Z_DIM"])(int_embed)
        z_int_std = nn.Dense(self.config["Z_DIM"])(int_embed)
        z_ext_mu = nn.Dense(self.config["Z_DIM"])(ext_embed)
        z_ext_std = nn.Dense(self.config["Z_DIM"])(ext_embed)

        return hidden, z_int_mu, nn.softplus(z_int_std), z_ext_mu, nn.softplus(z_ext_std)


class InfluenceDecoder(nn.Module):
    """One-step predictors. The direct head predicts the next value of the dynamic state
    coordinates from (state, own action, z); the response head predicts them from (state, y) only.
    Both heads output all `out_dim` dynamic coordinates; the partition masks select which
    coordinates each head is trained on (S^d for the direct head, S^r for the response head)."""
    config: Dict
    out_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, z_int, z_ext, state, action, training: bool = True):
        act = lambda x: nn.tanh(x)
        do = lambda x: nn.Dropout(rate=0.5, deterministic=not training)(x)
        norm = lambda x: nn.LayerNorm()(x)

        int_emb = nn.Dense(self.hidden_dim)(state)
        int_emb = act(int_emb)
        int_emb = do(int_emb)

        action_emb = nn.Dense(self.hidden_dim)(action)
        action_emb = act(action_emb)
        action_emb = do(action_emb)

        ext_emb = nn.Dense(self.hidden_dim)(state)
        ext_emb = act(ext_emb)
        ext_emb = do(ext_emb)

        z_int = norm(jnp.concatenate([z_int, int_emb, action_emb], axis=-1))
        z_ext = norm(jnp.concatenate([z_ext, ext_emb], axis=-1))

        h1 = act(nn.Dense(self.hidden_dim)(z_int))
        h2 = act(nn.Dense(self.hidden_dim)(z_ext))

        int_pred = nn.Dense(self.out_dim)(h1)
        ext_pred = nn.Dense(self.out_dim)(h2)

        return int_pred, ext_pred


class MLP(nn.Module):
    """Small MLP used for the auxiliary one-step models of the automatic partition."""
    out_dim: int
    hidden: int = 128
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.layers):
            x = nn.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(self.out_dim)(x)
