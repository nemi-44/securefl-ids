"""Drive the live AWS campaign from a workstation using the Terraform outputs.
Credentials come from the standard AWS environment; nothing is written to disk."""
import io, json, os, subprocess, sys, tarfile, time, numpy as np, boto3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from securefl.data import load, partition

R = "eu-west-1"; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tf = json.loads(subprocess.check_output(["terraform", "-chdir=" + ROOT + "/terraform", "output", "-json"]))
B, IID = tf["bucket"]["value"], tf["instance_id"]["value"]
s3, ssm = boto3.client("s3", region_name=R), boto3.client("ssm", region_name=R)
OUT = os.path.join(ROOT, "results", "cloud"); os.makedirs(OUT, exist_ok=True)
ARMS = {"fedavg": (0.0, 0.0), "fedavg_topk": (0.0, 0.1), "securefl": (1.0, 0.1)}
KS, SEEDS = [int(k) for k in os.environ.get("KS", "5,10").split(",")], range(int(os.environ.get("NSEEDS", 3)))


def npz(**a):
    b = io.BytesIO(); np.savez(b, **a); return b.getvalue()


def sh(cmd, timeout=1800):
    cid = ssm.send_command(InstanceIds=[IID], DocumentName="AWS-RunShellScript",
                           Parameters={"commands": [cmd], "executionTimeout": [str(timeout)]})["Command"]["CommandId"]
    while True:
        time.sleep(3)
        try: inv = ssm.get_command_invocation(CommandId=cid, InstanceId=IID)
        except ssm.exceptions.InvocationDoesNotExist: continue
        if inv["Status"] not in ("Pending", "InProgress", "Delayed"):
            return inv["Status"], inv["StandardOutputContent"][-300:] + inv["StandardErrorContent"][-300:]


def setup():
    Xtr, ytr, ctr, Xte, yte, cte = load(ROOT + "/data/UNSW_NB15_training-set.csv", ROOT + "/data/UNSW_NB15_testing-set.csv")
    s3.put_object(Bucket=B, Key="eval/test.npz", Body=npz(X=Xte, y=yte, cats=cte.astype(str)))
    for k in KS:
        for s in SEEDS:
            parts = partition(ctr, k, np.random.default_rng(1000 + s), 0.5)
            for j, p in enumerate(parts):
                s3.put_object(Bucket=B, Key=f"shards/{k}/{s}/c{j}.npz", Body=npz(X=Xtr[p], y=ytr[p]))
            s3.put_object(Bucket=B, Key=f"shards/{k}/{s}/meta.json", Body=json.dumps(dict(sizes=[len(p) for p in parts])).encode())
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        t.add(ROOT + "/securefl", "securefl", filter=lambda x: None if "__pycache__" in x.name else x)
        for f in ["cloud_node.py", "Dockerfile", "node_run.sh"]: t.add(ROOT + "/cloud/" + f, f)
    s3.put_object(Bucket=B, Key="code/code.tgz", Body=b.getvalue())
    print("build", sh(f"mkdir -p /opt/sfl && cd /opt/sfl && aws s3 cp s3://{B}/code/code.tgz . --region {R} --quiet && "
                      "tar xzf code.tgz && docker build -q -t securefl . | tail -1"))


def campaign():
    for s in SEEDS:
        for k in KS:
            for arm, (z, ratio) in ARMS.items():
                run = f"{arm}_k{k}_s{s}"
                if os.path.exists(f"{OUT}/{run}.json"): continue
                st, out = sh(f"bash /opt/sfl/node_run.sh {B} {run} {arm} {k} {s} {z} {ratio}")
                try:
                    res = json.loads(s3.get_object(Bucket=B, Key=f"results/{run}.json")["Body"].read())
                    res["docker_stats_csv"] = s3.get_object(Bucket=B, Key=f"stats/{run}.csv")["Body"].read().decode()
                    json.dump(res, open(f"{OUT}/{run}.json", "w"))
                    print(run, st, round(res["test"]["f1"], 4), round(res["wall_s"], 1), flush=True)
                except Exception as e:
                    print(run, st, "FAILED", out.replace("\n", " | ")[-200:], flush=True)


if __name__ == "__main__":
    {"setup": setup, "campaign": campaign}[sys.argv[1]]()
