"""Aggregate local and cloud results: seed-level means, 95% t-intervals,
paired t-tests with Holm correction, LaTeX tables and 300 dpi figures."""
import glob, json, os, re, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.environ.get("REPORT_DIR", os.path.join(ROOT, "report_assets"))
os.makedirs(f"{REP}/tables", exist_ok=True); os.makedirs(f"{REP}/figures", exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
PAL = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8d99ae"]

L = {}
for f in glob.glob(f"{ROOT}/results/local/*.json"):
    r = json.load(open(f)); L.setdefault(r["config"]["name"], {})[r["config"]["seed"]] = r


def vals(arm, key="f1", src=None):
    src = src or L
    return np.array([(r["test"][key] if key in r["test"] else r[key]) for _, r in sorted(src[arm].items())], float)


def ci(x):
    m = x.mean(); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - h, m + h


def fmt(x, d=3):
    m, lo, hi = ci(x); return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j]); adj[j] = min(1, run)
    return adj


def tex(name, head, rows, spec):
    s = "\\begin{tabular}{" + spec + "}\n\\toprule\n" + " & ".join(head) + " \\\\\n\\midrule\n"
    s += "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n"
    open(f"{REP}/tables/{name}.tex", "w").write(s)


S = {}
MAIN = [("central", "Centralised MLP"), ("central_dp", "Centralised DP-SGD"), ("fedavg", "FedAvg"),
        ("fedavg_dp", "FedAvg + DP-SGD"), ("fedavg_topk", "FedAvg + top-k"), ("securefl", "SecureFL-IDS")]
rows = []
for a, lab in MAIN:
    eps = [r["epsilon"] for r in L[a].values() if r.get("epsilon")]
    rows.append([lab, fmt(vals(a, "accuracy")), fmt(vals(a, "precision")), fmt(vals(a, "recall")), fmt(vals(a, "f1")),
                 fmt(vals(a, "fpr")), fmt(vals(a, "auc")), f"{np.mean(vals(a, 'up_mb')):.1f}",
                 f"{np.median(eps):.1f}" if eps else "n/a"])
    S[a] = {k: ci(vals(a, k)) for k in ["accuracy", "precision", "recall", "f1", "fpr", "auc", "up_mb", "train_s"]}
    S[a]["n"] = len(L[a]); S[a]["eps_median"] = float(np.median(eps)) if eps else None
    S[a]["eps_range"] = [float(min(eps)), float(max(eps))] if eps else None
tex("main", ["Arm", "Accuracy", "Precision", "Recall", "F1", "FPR", "AUC", "Up MB", "$\\varepsilon$"],
    [[r[0]] + [c.replace(" [", "\\newline{\\scriptsize [").replace("]", "]}") if "[" in c else c for c in r[1:]] for r in rows],
    "l" + "p{1.3cm}" * 6 + "rr")

PAIRS = [("securefl", "fedavg"), ("securefl", "central"), ("fedavg", "central"), ("fedavg_dp", "fedavg"),
         ("fedavg_topk", "fedavg"), ("securefl", "fedavg_dp"), ("central_dp", "central")]
names = dict(MAIN); res = []
for metric in ["f1", "auc"]:
    ps, out = [], []
    for a, b in PAIRS:
        d = vals(a, metric) - vals(b, metric); t, p = stats.ttest_rel(vals(a, metric), vals(b, metric))
        m, lo, hi = ci(d); ps.append(p); out.append([a, b, m, lo, hi, p])
    for o, ph in zip(out, holm(ps)): o.append(ph); res.append([metric] + o)
S["pairs"] = [dict(metric=r[0], a=r[1], b=r[2], diff=r[3], lo=r[4], hi=r[5], p=r[6], p_holm=r[7]) for r in res]
tex("pairs", ["Metric", "Comparison", "Mean diff. [95\\% CI]", "$p$", "$p_{\\mathrm{Holm}}$"],
    [[r[0].upper(), f"{names[r[1]]} $-$ {names[r[2]]}", f"{r[3]:+.4f} [{r[4]:+.4f}, {r[5]:+.4f}]",
      f"{r[6]:.2g}", f"{r[7]:.2g}"] for r in res], "llrrr")

# sweeps
def sweep(prefixes):
    return [(k, ci(vals(k, "f1")), ci(vals(k, "up_mb")), [r.get("epsilon") for r in L[k].values()]) for k in prefixes if k in L]
zs = [0.5, 0.7, 1.0, 1.5, 2.0]
zarm = {z: ("securefl" if z == 1.0 else f"sweep_z{z}") for z in zs}
S["z_sweep"] = {str(z): dict(f1=ci(vals(zarm[z])), eps_median=float(np.median([r["epsilon"] for r in L[zarm[z]].values()])),
                             eps_max=float(max(r["epsilon"] for r in L[zarm[z]].values()))) for z in zs}
rs = [0.01, 0.05, 0.1, 0.25]
rsec = {r: ("securefl" if r == 0.1 else f"sweep_k{r}_sec") for r in rs}; rno = {r: ("fedavg_topk" if r == 0.1 else f"sweep_k{r}_nodp") for r in rs}
S["k_sweep"] = {str(r): dict(sec=ci(vals(rsec[r])), nodp=ci(vals(rno[r])), up_mb=float(np.mean(vals(rno[r], "up_mb")))) for r in rs}
Ks = [5, 10, 20, 40]
karm = lambda k, n: n if k == 10 else f"scale_c{k}_{n}"
S["scale"] = {str(k): {n: dict(f1=ci(vals(karm(k, n))), up_mb=float(np.mean(vals(karm(k, n), "up_mb"))),
                               train_s=ci(vals(karm(k, n), "train_s"))) for n in ["fedavg", "securefl"]} for k in Ks}
As = ["0", "1.0", "0.5", "0.1"]
aarm = lambda a, n: n if a == "0.5" else f"noniid_a{a}_{n}"
S["noniid"] = {a: {n: ci(vals(aarm(a, n))) for n in ["fedavg", "securefl"]} for a in As}
rows = [[f"{z}", f"{S['z_sweep'][str(z)]['eps_median']:.1f} ({S['z_sweep'][str(z)]['eps_max']:.1f})", fmt(vals(zarm[z]))] for z in zs]
tex("zsweep", ["Noise multiplier $z$", "$\\varepsilon$ median (max)", "F1 [95\\% CI]"], rows, "rrr")
rows = [[f"{int(r*100)}\\%", f"{S['k_sweep'][str(r)]['up_mb']:.1f}", fmt(vals(rno[r])), fmt(vals(rsec[r]))] for r in rs]
rows.append(["100\\% (dense)", f"{np.mean(vals('fedavg','up_mb')):.1f}", fmt(vals("fedavg")), fmt(vals("fedavg_dp"))])
tex("ksweep", ["Kept coords", "Upload MB", "F1 no DP", "F1 with DP ($z{=}1$)"], rows, "rrrr")
rows = [[str(k), f"{S['scale'][str(k)]['fedavg']['up_mb']:.1f}", fmt(vals(karm(k, 'fedavg'))),
         f"{S['scale'][str(k)]['securefl']['up_mb']:.1f}", fmt(vals(karm(k, 'securefl')))] for k in Ks]
tex("scale", ["Clients", "FedAvg MB", "FedAvg F1", "SecureFL MB", "SecureFL F1"], rows, "rrrrr")
rows = [["IID" if a == "0" else f"Dir($\\alpha{{=}}{a}$)", fmt(vals(aarm(a, 'fedavg'))), fmt(vals(aarm(a, 'securefl')))] for a in As]
tex("noniid", ["Partition", "FedAvg F1", "SecureFL-IDS F1"], rows, "lrr")


# exploratory clipping analysis (added after the noise sweep showed little effect of z)
CL = [("fedavg_topk", "none", "0", "no"), ("fedavg_cliponly", "1.0", "0", "no")] + \
     [(f"sweep_clip{c}" if c != "1.0" else "securefl", c, "1.0", "yes") for c in ["0.3", "1.0", "3.0", "10.0"]]
CL = [c for c in CL if c[0] in L]
S["clip"] = {a: dict(f1=ci(vals(a)), auc=ci(vals(a, "auc")), fpr=ci(vals(a, "fpr"))) for a, *_ in CL}
tex("clip", ["Clip $C$", "Noise $z$", "DP", "F1 [95\\% CI]", "AUC [95\\% CI]", "FPR"],
    [[c, z, d, fmt(vals(a)), fmt(vals(a, "auc")), f"{np.mean(vals(a, 'fpr')):.3f}"] for a, c, z, d in CL], "rrrrrr")
if "fedavg_cliponly" in L:
    ex = []
    for a, b in [("fedavg_cliponly", "fedavg_topk"), ("securefl", "fedavg_cliponly")]:
        d = vals(a) - vals(b); t, p = stats.ttest_rel(vals(a), vals(b)); ex.append([a, b, *ci(d), p])
    for e, ph in zip(ex, holm([e[-1] for e in ex])): e.append(ph)
    S["clip_pairs"] = [dict(a=e[0], b=e[1], diff=e[2], lo=e[3], hi=e[4], p=e[5], p_holm=e[6]) for e in ex]

# per-category detection rate (test set) for main arms
cats = sorted(next(iter(L["central"].values()))["test"]["per_cat_detect"])
S["per_cat"] = {a: {c: float(np.mean([r["test"]["per_cat_detect"][c] for r in L[a].values()])) for c in cats} for a, _ in MAIN}

# figures
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
for i, (a, lab) in enumerate(MAIN):
    m, lo, hi = S[a]["f1"]; ax[0].bar(i, m, color=PAL[i], yerr=[[m - lo], [hi - m]], capsize=2)
    x = vals(a); ax[0].scatter(np.full(len(x), i) + np.linspace(-.2, .2, len(x)), x, s=3, c="k", zorder=3)
ax[0].set_xticks(range(6)); ax[0].set_xticklabels(["Central", "Central\nDP", "FedAvg", "FedAvg\n+DP", "FedAvg\n+top-k", "SecureFL\nIDS"], fontsize=6)
ax[0].set_ylim(0.78, 0.92); ax[0].set_ylabel("Test F1 (10 seeds)")
for i, (a, lab) in enumerate(MAIN):
    if a.startswith("central"): continue
    h = np.array([[p["f1"] for p in r["history"]] for r in L[a].values()]); rr = [p["round"] for p in next(iter(L[a].values()))["history"]]
    ax[1].plot(rr, h.mean(0), "-o", ms=2, color=PAL[i], label=lab)
ax[1].set_xlabel("Round"); ax[1].set_ylabel("Mean test F1"); ax[1].legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(f"{REP}/figures/main_f1.png", dpi=300); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.2))
e = [S["z_sweep"][str(z)]["eps_median"] for z in zs]; f = [S["z_sweep"][str(z)]["f1"] for z in zs]
ax[0].errorbar(e, [x[0] for x in f], yerr=[[x[0] - x[1] for x in f], [x[2] - x[0] for x in f]], fmt="o-", color=PAL[4], ms=3, capsize=2)
ax[0].axhline(S["fedavg_topk"]["f1"][0], ls="--", c="grey", lw=.8); ax[0].set_xscale("log")
ax[0].set_xlabel("$\\varepsilon$ (median, $\\delta=10^{-5}$)"); ax[0].set_ylabel("F1"); ax[0].set_title("(a) privacy vs utility", fontsize=7)
for key, c, lab in [("nodp", PAL[1], "no DP"), ("sec", PAL[4], "DP $z$=1")]:
    xs = [S["k_sweep"][str(r)]["up_mb"] for r in rs]; ys = [S["k_sweep"][str(r)][key][0] for r in rs]
    ax[1].plot(xs, ys, "o-", ms=3, color=c, label=lab)
ax[1].scatter([np.mean(vals("fedavg", "up_mb"))], [S["fedavg"]["f1"][0]], c=PAL[0], marker="s", s=10, label="dense FedAvg")
ax[1].set_xscale("log"); ax[1].set_xlabel("Upload MB per run"); ax[1].set_title("(b) communication vs F1", fontsize=7); ax[1].legend(fontsize=5, frameon=False)
for n, c in [("fedavg", PAL[0]), ("securefl", PAL[4])]:
    ax[2].plot(Ks, [S["scale"][str(k)][n]["f1"][0] for k in Ks], "o-", ms=3, color=c, label=n)
ax[2].set_xscale("log", base=2); ax[2].set_xticks(Ks); ax[2].set_xticklabels(Ks); ax[2].set_xlabel("Clients"); ax[2].set_title("(c) number of clients", fontsize=7)
ax[2].legend(fontsize=5, frameon=False)
fig.tight_layout(); fig.savefig(f"{REP}/figures/sweeps.png", dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 2.0)); w = 0.13
for i, (a, lab) in enumerate(MAIN):
    ax.bar(np.arange(len(cats)) + (i - 2.5) * w, [S["per_cat"][a][c] for c in cats], w, color=PAL[i], label=lab)
ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=6); ax.set_ylabel("Flagged as attack")
ax.legend(fontsize=5, ncol=3, frameon=False, loc="lower left"); fig.tight_layout(); fig.savefig(f"{REP}/figures/per_cat.png", dpi=300); plt.close(fig)

# cloud
C = {}
for f in glob.glob(f"{ROOT}/results/cloud/*.json"):
    r = json.load(open(f)); C.setdefault((r["arm"], r["k"]), []).append(r)
if C:
    def stat_rows(rs):
        cpu, mem = [], []
        for r in rs:
            for line in r["docker_stats_csv"].splitlines():
                p = line.split(",")
                if len(p) < 3 or not p[0].startswith("c"): continue
                cpu.append(float(p[1].rstrip("%")))
                m = re.match(r"([\d.]+)(\w+)", p[2]); mul = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}.get(m.group(2), 1); mem.append(float(m.group(1)) * mul)
        return (np.mean(cpu) if cpu else np.nan, np.percentile(mem, 95) if mem else np.nan)
    rows, S["cloud"] = [], {}
    for (arm, k) in sorted(C, key=lambda x: (x[1], ["fedavg", "fedavg_topk", "securefl"].index(x[0]))):
        rs = C[(arm, k)]; f1 = np.array([r["test"]["f1"] for r in rs]); wall = np.array([r["wall_s"] for r in rs])
        rt = np.concatenate([r["round_s"] for r in rs]); up = np.array([r["upload_bytes_s3"] for r in rs]) / 1e6
        cpu, mem = stat_rows(rs)
        S["cloud"][f"{arm}_k{k}"] = dict(n=len(rs), f1=ci(f1), wall=ci(wall), round_med=float(np.median(rt)),
                                          round_p95=float(np.percentile(rt, 95)), up_mb=float(up.mean()), cpu=cpu, mem_p95=mem)
        rows.append([{"fedavg": "FedAvg", "fedavg_topk": "FedAvg + top-k", "securefl": "SecureFL-IDS"}[arm], str(k), str(len(rs)),
                     fmt(f1), f"{wall.mean():.0f}", f"{np.median(rt):.2f} / {np.percentile(rt, 95):.2f}", f"{up.mean():.2f}", f"{cpu:.0f}", f"{mem:.0f}"])
    tex("cloud", ["Arm", "$K$", "Runs", "F1 [95\\% CI]", "Wall s", "Round s med/p95", "S3 MB", "CPU \\%", "Mem MiB"], rows, "lrrrrrrrr")
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for i, key in enumerate(S["cloud"]):
        rt = np.concatenate([r["round_s"] for r in C[(key.rsplit("_k", 1)[0], int(key.rsplit("_k", 1)[1]))]])
        ax.boxplot(rt, positions=[i], widths=.5, showfliers=False)
    ax.set_xticks(range(len(S["cloud"]))); ax.set_xticklabels([k.replace("fedavg_topk", "top-k").replace("_k", "\nK=") for k in S["cloud"]], fontsize=5)
    ax.set_ylabel("Round time on AWS (s)"); fig.tight_layout(); fig.savefig(f"{REP}/figures/cloud_rounds.png", dpi=300); plt.close(fig)

json.dump(S, open(f"{ROOT}/results/summary.json", "w"), indent=1, default=float)
print("analysis ok", len(L), "local arms,", sum(len(v) for v in C.values()), "cloud runs")
