"""Federated averaging with optional DP-SGD clients and top-k sparsified,
fp16-quantised uploads with client-side error feedback."""
import time, numpy as np

from .model import MLP, local_train


def topk(u, ratio):
    """Return (indices, fp16 values) of the largest-magnitude ratio*|u| entries."""
    k = max(1, int(ratio * u.size))
    idx = np.argpartition(np.abs(u), -k)[-k:].astype(np.int32)
    return idx, u[idx].astype(np.float16)


def upload_bytes(size, ratio):
    return 4 * size if ratio is None else 6 * max(1, int(ratio * size))  # int32 idx + fp16 value


def metrics(y, p, cats=None, thr=0.5):
    from sklearn.metrics import roc_auc_score
    yh = (p >= thr).astype(int); y = y.astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
    tn = int(((yh == 0) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    out = dict(accuracy=(tp + tn) / len(y), precision=pr, recall=rc,
               f1=2 * pr * rc / max(pr + rc, 1e-12), fpr=fp / max(fp + tn, 1),
               auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"))
    if cats is not None:
        out["per_cat_detect"] = {str(c): float(yh[cats == c].mean()) for c in np.unique(cats)}
    return out


def run_fl(Xtr, ytr, parts, Xte, yte, seed, rounds=30, epochs=1, lr=0.05, batch=256,
           clip=None, z=0.0, ratio=None, transport=None, log=None):
    """transport(round, client, payload)->None lets the cloud runner push
    payloads through S3; locally payloads stay in memory."""
    rng = np.random.default_rng(seed); m = MLP(Xtr.shape[1]); w = m.init(rng)
    resid = [np.zeros_like(w) for _ in parts]; sizes = np.array([len(p) for p in parts], float)
    up = down = 0; t0 = time.time(); hist = []; steps = []
    for r in range(rounds):
        agg = np.zeros_like(w)
        for j, p in enumerate(parts):
            wj, st = local_train(m, w, Xtr[p], ytr[p], rng, epochs, lr, batch, clip, z)
            steps.append((j, len(p), st)); u = wj - w
            if ratio is not None:
                u = u + resid[j]; idx, val = topk(u, ratio)
                sent = np.zeros_like(u); sent[idx] = val.astype(np.float32); resid[j] = u - sent; u = sent
            if transport: transport(r, j, u)
            agg += sizes[j] / sizes.sum() * u
            up += upload_bytes(w.size, ratio); down += 4 * w.size
        w = w + agg
        if log is not None and (r % 5 == 4 or r == rounds - 1):
            hist.append(dict(round=r + 1, f1=metrics(yte, m.predict(w, Xte))["f1"]))
    return w, m, dict(train_s=time.time() - t0, up_mb=up / 1e6, down_mb=down / 1e6,
                      params=int(w.size), history=hist, steps=steps)


def run_central(Xtr, ytr, Xte, yte, seed, epochs=30, lr=0.05, batch=256, clip=None, z=0.0):
    rng = np.random.default_rng(seed); m = MLP(Xtr.shape[1]); w = m.init(rng); t0 = time.time()
    w, st = local_train(m, w, Xtr, ytr, rng, epochs, lr, batch, clip, z)
    return w, m, dict(train_s=time.time() - t0, up_mb=0.0, down_mb=0.0, params=int(w.size),
                      steps=[(0, len(ytr), st)])


def epsilon(n, batch, steps, z, delta=1e-5):
    """RDP accountant (dp-accounting) for Poisson-subsampled Gaussian DP-SGD."""
    if not z: return float("inf")
    from dp_accounting import rdp, dp_event as e
    a = rdp.RdpAccountant()
    a.compose(e.PoissonSampledDpEvent(min(1.0, batch / n), e.GaussianDpEvent(z)), steps)
    return float(a.get_epsilon(delta))
