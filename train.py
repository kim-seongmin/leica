"""Train LEICA on SMAX with exogenous opponent regimes.

    python train.py                                   # default configuration (leica/config.yaml)
    python train.py MAP_NAME=10m_vs_11m SEED=3        # KEY=VALUE overrides (nested keys: ENV_KWARGS.see_enemy_actions=false)
    python train.py --config my_config.yaml

Outputs (SAVE_PATH/<ALG_NAME>_<MAP_NAME>_<TASK>_seed<SEED>/):
    config.yaml       the configuration that was run
    metrics.npz       per-update training/evaluation metrics, shape [NUM_SEEDS, NUM_UPDATES]
    summary.json      final greedy win rates (seen / unseen regimes), the estimated partition and influence weights
    params_<k>.msgpack  actor and context-encoder parameters of seed k (flax serialization)
"""
import argparse
import copy
import json
import os
import time

import jax
import numpy as np
import yaml
from flax import serialization

from leica.leica import make_train


def parse_value(v: str):
    try:
        return json.loads(v)  # numbers, booleans, null, lists
    except json.JSONDecodeError:
        return v


def apply_override(config: dict, key: str, value):
    node = config
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "leica", "config.yaml"))
    ap.add_argument("overrides", nargs="*", help="KEY=VALUE overrides of the configuration")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    for kv in args.overrides:
        k, v = kv.split("=", 1)
        apply_override(config, k, parse_value(v))
    print("Config:\n" + yaml.safe_dump(config, sort_keys=False), flush=True)

    train = make_train(copy.deepcopy(config))  # make_train adds derived (non-yaml) entries to its copy
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(train))
    t0 = time.time()
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"training finished in {time.time() - t0:.1f}s", flush=True)

    metrics = jax.tree.map(np.asarray, outs["metrics"])
    train_state = outs["runner_state"][0]
    D = np.asarray(train_state.D)
    w = np.asarray(train_state.w_ex)
    chi = np.asarray(train_state.chi)
    seen = np.asarray(outs["final_test"]["seen"]["returned_won_episode"])
    unseen = np.asarray(outs["final_test"]["unseen"]["returned_won_episode"])
    for k in range(config["NUM_SEEDS"]):
        direct = [nm for nm, m in zip(train.coord_names, D[k]) if m]
        print(f"seed {k}: final greedy win rate seen {seen[k]:.3f} unseen {unseen[k]:.3f}; "
              f"S^d = {len(direct)}/{D.shape[1]} coordinates: {', '.join(direct)}")

    if config.get("SAVE_PATH"):
        run_name = f'{config["ALG_NAME"]}_{config["MAP_NAME"]}_{config.get("TASK", "")}_seed{config["SEED"]}'
        out_dir = os.path.join(config["SAVE_PATH"], run_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        np.savez(os.path.join(out_dir, "metrics.npz"), **{k.replace("/", "_"): v for k, v in metrics.items()})
        summary = {
            "seen_win_rate": seen.tolist(),
            "unseen_win_rate": unseen.tolist(),
            "coordinates": train.coord_names,
            "direct_mask": D.astype(int).tolist(),
            "partition_score": chi.tolist(),
            "influence_weights": w.tolist(),
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        for k in range(config["NUM_SEEDS"]):
            params = jax.tree.map(lambda x: np.asarray(x)[k], {m: train_state.params[m] for m in ("actor", "encoder")})
            with open(os.path.join(out_dir, f"params_{k}.msgpack"), "wb") as f:
                f.write(serialization.to_bytes(params))
        print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
