"""Write report_assets/macros.tex: every number quoted in prose that is not
inside a generated table is defined here from results/summary.json."""
import json, os, glob, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.environ.get("REPORT_DIR", os.path.join(ROOT, "report_assets"))
S = json.load(open(f"{ROOT}/results/summary.json"))
m = {}
cp = {(p["a"], p["b"]): p for p in S.get("clip_pairs", [])}
m["CLIPPH"] = f"{cp[('securefl', 'fedavg_cliponly')]['p_holm']:.2f}"
m["CLIPDIFF"] = f"{cp[('fedavg_cliponly', 'fedavg_topk')]['diff']:+.3f}".replace("-", "$-$")
m["CLIPTENF"] = f"{S['clip']['sweep_clip10.0']['f1'][0]:.3f}"
eps = lambda a: [json.load(open(f))["epsilon"] for f in glob.glob(f"{ROOT}/results/local/{a}_s*.json")]
m["EPSFIVE"] = f"{np.median(eps('scale_c5_securefl')):.1f}"; m["EPSFORTY"] = f"{np.median(eps('scale_c40_securefl')):.1f}"
m["EPSMAIN"] = f"{np.median(eps('securefl')):.1f}"; m["EPSCENTRAL"] = f"{np.median(eps('central_dp')):.1f}"
m["EPSMAINMAX"] = f"{max(eps('securefl')):.1f}"
for k, v in json.load(open(f"{ROOT}/results/macros_extra.json")).items() if os.path.exists(f"{ROOT}/results/macros_extra.json") else []:
    m[k] = v
with open(f"{REP}/macros.tex", "w") as f:
    for k, v in m.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("macros", len(m))
