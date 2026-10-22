"""Container entrypoint for the live AWS run. Each client container holds only
its own shard; the server container never sees training rows. Model state is
exchanged through the S3 parameter bus; timings go to CloudWatch."""
import io, json, os, sys, time, numpy as np, boto3
sys.path.insert(0, "/app")
from securefl.model import MLP, local_train
from securefl.fl import topk, metrics

B, RUN, ARM = os.environ["BUCKET"], os.environ["RUN"], os.environ["ARM"]
K, SEED, ROUNDS = int(os.environ["K"]), int(os.environ["SEED"]), int(os.environ.get("ROUNDS", 30))
Z = float(os.environ.get("Z", 0)); RATIO = float(os.environ.get("RATIO", 0)) or None
s3 = boto3.client("s3", region_name="eu-west-1"); cw = boto3.client("cloudwatch", region_name="eu-west-1")


def put(key, arr_dict):
    b = io.BytesIO(); np.savez(b, **arr_dict); s3.put_object(Bucket=B, Key=key, Body=b.getvalue())
    return b.tell()


def get(key, wait=True):
    while True:
        try: return np.load(io.BytesIO(s3.get_object(Bucket=B, Key=key)["Body"].read()))
        except s3.exceptions.NoSuchKey:
            if not wait: return None
            time.sleep(0.2)


def metric(name, v, unit):
    cw.put_metric_data(Namespace="SecureFL-IDS", MetricData=[dict(
        MetricName=name, Value=float(v), Unit=unit, Dimensions=[dict(Name="Arm", Value=ARM)])])


def client(j):
    d = get(f"shards/{K}/{SEED}/c{j}.npz"); X, y = d["X"], d["y"]
    m = MLP(X.shape[1]); rng = np.random.default_rng(SEED * 1000 + j); resid = None; steps = 0
    for r in range(ROUNDS):
        w = get(f"runs/{RUN}/global/r{r}.npz")["w"]
        wj, st = local_train(m, w, X, y, rng, clip=1.0 if Z else None, z=Z); steps += st
        u = wj - w
        if RATIO:
            u = u + (resid if resid is not None else 0); idx, val = topk(u, RATIO)
            sent = np.zeros_like(u); sent[idx] = val.astype(np.float32); resid = u - sent
            nb = put(f"runs/{RUN}/upd/r{r}/c{j}.npz", dict(idx=idx, val=val))
        else:
            nb = put(f"runs/{RUN}/upd/r{r}/c{j}.npz", dict(u=u))
        if j == 0: metric("UploadBytes", nb * K, "Bytes")
    put(f"runs/{RUN}/steps/c{j}.npz", dict(n=np.array(len(y)), steps=np.array(steps)))


def server():
    meta = json.loads(s3.get_object(Bucket=B, Key=f"shards/{K}/{SEED}/meta.json")["Body"].read())
    sizes = np.array(meta["sizes"], float); wts = sizes / sizes.sum()
    te = get("eval/test.npz"); m = MLP(te["X"].shape[1]); w = m.init(np.random.default_rng(SEED))
    t0 = time.time(); rt = []; up = 0
    put(f"runs/{RUN}/global/r0.npz", dict(w=w))
    for r in range(ROUNDS):
        ts = time.time(); agg = np.zeros_like(w)
        for j in range(K):
            key = f"runs/{RUN}/upd/r{r}/c{j}.npz"; d = get(key)
            up += s3.head_object(Bucket=B, Key=key)["ContentLength"]
            if "u" in d: agg += wts[j] * d["u"]
            else:
                u = np.zeros_like(w); u[d["idx"]] = d["val"].astype(np.float32); agg += wts[j] * u
        w = w + agg; put(f"runs/{RUN}/global/r{r + 1}.npz", dict(w=w))
        rt.append(time.time() - ts); metric("RoundSeconds", rt[-1], "Seconds")
    res = dict(run=RUN, arm=ARM, k=K, seed=SEED, rounds=ROUNDS, z=Z, ratio=RATIO,
               wall_s=time.time() - t0, round_s=rt, upload_bytes_s3=up,
               test=metrics(te["y"], m.predict(w, te["X"]), te["cats"]))
    s3.put_object(Bucket=B, Key=f"results/{RUN}.json", Body=json.dumps(res).encode())


if __name__ == "__main__":
    server() if os.environ["ROLE"] == "server" else client(int(os.environ["CID"]))
