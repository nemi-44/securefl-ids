"""Print the full local experiment matrix (one command per line)."""
S = range(10)
B = "--clients 10 --alpha 0.5 --rounds 30"
arms = {"central": "--mode central", "central_dp": "--mode central --z 1.0",
        "fedavg": B, "fedavg_dp": B + " --z 1.0", "fedavg_topk": B + " --ratio 0.1",
        "securefl": B + " --z 1.0 --ratio 0.1"}
arms["fedavg_cliponly"] = B + " --cliponly --ratio 0.1"
for c in [0.3, 3.0, 10.0]: arms[f"sweep_clip{c}"] = B + f" --z 1.0 --clip {c} --ratio 0.1"
for z in [0.5, 0.7, 1.5, 2.0]: arms[f"sweep_z{z}"] = B + f" --z {z} --ratio 0.1"
for r in [0.01, 0.05, 0.25]:
    arms[f"sweep_k{r}_sec"] = B + f" --z 1.0 --ratio {r}"; arms[f"sweep_k{r}_nodp"] = B + f" --ratio {r}"
for k in [5, 20, 40]:
    for n, x in [("fedavg", ""), ("securefl", " --z 1.0 --ratio 0.1")]:
        arms[f"scale_c{k}_{n}"] = f"--clients {k} --alpha 0.5 --rounds 30" + x
for al in [0, 1.0, 0.1]:
    for n, x in [("fedavg", ""), ("securefl", " --z 1.0 --ratio 0.1")]:
        arms[f"noniid_a{al}_{n}"] = f"--clients 10 --alpha {al} --rounds 30" + x
for n, c in arms.items():
    for s in S: print(f"python scripts/run_local.py --name {n} --seed {s} {c}")
