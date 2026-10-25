"""Independent checks on the packaged results. Exits non-zero on any failure."""
import glob, json, os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ok = fail = 0


def check(cond, msg):
    global ok, fail
    if cond: ok += 1
    else: fail += 1; print("FAIL:", msg)


sys.path.insert(0, ROOT)
L = [json.load(open(f)) for f in glob.glob(f"{ROOT}/results/local/*.json")]
import subprocess
N = len(subprocess.check_output([sys.executable, f"{ROOT}/scripts/matrix.py"]).splitlines())
check(len(L) == N, f"expected {N} local runs, found {len(L)}")
S = json.load(open(f"{ROOT}/results/summary.json"))
arms = {}
for r in L: arms.setdefault(r["config"]["name"], []).append(r)
check(len(arms) == N // 10, "all local configurations present")
check(all(len(v) == 10 for v in arms.values()), "10 seeds per configuration")
for a in ["central", "central_dp", "fedavg", "fedavg_dp", "fedavg_topk", "securefl"]:
    f1 = np.mean([r["test"]["f1"] for r in arms[a]])
    check(abs(f1 - S[a]["f1"][0]) < 1e-9, f"summary F1 for {a} matches raw runs")
    for r in arms[a]:
        t = r["test"]; check(0 <= t["fpr"] <= 1 and 0.5 <= t["auc"] <= 1, f"{a} metric ranges")
        check((r["epsilon"] is not None) == (r["config"]["z"] > 0), f"{a} epsilon reported iff DP")
for r in L:
    c = r["config"]
    if c["mode"] == "fl":
        P = r["params"]; per = 4 * P if not c["ratio"] else 6 * int(c["ratio"] * P)
        check(abs(r["up_mb"] - per * c["clients"] * c["rounds"] / 1e6) < 1e-6, f"{c['name']} upload bytes formula")
# ranking claim used in the report: sorted by mean F1
rank = sorted(["central", "central_dp", "fedavg", "fedavg_dp", "fedavg_topk", "securefl"], key=lambda a: -S[a]["f1"][0])
json.dump(rank, open(f"{ROOT}/results/ranking.json", "w"))
for p in S["pairs"]: check(p["p_holm"] >= p["p"] - 1e-12, "Holm-adjusted p >= raw p")
C = [json.load(open(f)) for f in glob.glob(f"{ROOT}/results/cloud/*.json")]
check(len(C) >= 3, "cloud results present")
for r in C:
    check(len(r["round_s"]) == r["rounds"], f"{r['run']} has one timing per round")
    check(r["docker_stats_csv"].count("\n") > 0, f"{r['run']} has container stats")
for f in glob.glob(f"{ROOT}/**/*", recursive=True):
    if os.path.isfile(f) and os.path.getsize(f) < 5e6 and not f.endswith(".csv"):
        check(("AK" + "IA").encode() not in open(f, "rb").read(), f"no access key id in {f}")
check(not glob.glob(f"{ROOT}/**/*.tfstate*", recursive=True), "no terraform state packaged")
print(f"verify: {ok} passed, {fail} failed"); sys.exit(1 if fail else 0)
