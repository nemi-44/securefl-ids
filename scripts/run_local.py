"""Run one local experiment configuration and write results/local/<name>.json."""
import logging; logging.disable(logging.WARNING)
import argparse, json, os, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from securefl.data import load, partition
from securefl.fl import run_fl, run_central, metrics, epsilon

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--mode", default="fl")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--clients", type=int, default=10)
ap.add_argument("--alpha", type=float, default=0.5)  # <=0 means IID
ap.add_argument("--rounds", type=int, default=30); ap.add_argument("--z", type=float, default=0.0)
ap.add_argument("--clip", type=float, default=1.0); ap.add_argument("--ratio", type=float, default=0.0)
ap.add_argument("--out", default="results/local")
ap.add_argument("--cliponly", action="store_true")  # clip per example, no noise
a = ap.parse_args()
Xtr, ytr, ctr, Xte, yte, cte = load("data/UNSW_NB15_training-set.csv", "data/UNSW_NB15_testing-set.csv")
dp = a.z > 0; clip = a.clip if (dp or a.cliponly) else None; ratio = a.ratio or None
if a.mode == "central":
    w, m, info = run_central(Xtr, ytr, Xte, yte, a.seed, a.rounds, clip=clip, z=a.z)
else:
    parts = partition(ctr, a.clients, np.random.default_rng(1000 + a.seed), None if a.alpha <= 0 else a.alpha)
    w, m, info = run_fl(Xtr, ytr, parts, Xte, yte, a.seed, a.rounds, clip=clip, z=a.z, ratio=ratio, log=True)
eps = {}
for j, n, st in info["steps"]:  # accumulate DP-SGD steps per client
    eps[j] = (n, eps.get(j, (n, 0))[1] + st)
info["epsilon"] = max(epsilon(n, 256, st, a.z) for n, st in eps.values()) if dp else None
info.pop("steps")
res = dict(config=vars(a), **info, test=metrics(yte, m.predict(w, Xte), cte))
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/{a.name}_s{a.seed}.json", "w"), indent=1)
print(a.name, a.seed, round(res["test"]["f1"], 4), round(info["train_s"], 1), info["epsilon"])
