"""UNSW-NB15 loading and preprocessing. All statistics are fitted on the
official training partition only; the official testing partition is never
used for fitting, tuning or thresholding."""
import numpy as np, pandas as pd

CAT = ["proto", "service", "state"]
DROP = ["id", "attack_cat", "label"]


def load(train_csv, test_csv, top_proto=8):
    tr, te = pd.read_csv(train_csv), pd.read_csv(test_csv)
    num = [c for c in tr.columns if c not in DROP + CAT]
    vocab = {}
    for c in CAT:
        vc = tr[c].value_counts()
        keep = list(vc.index[:top_proto]) if c == "proto" else list(vc.index[vc >= 50])
        vocab[c] = keep

    def enc(df):
        X = np.log1p(np.clip(df[num].to_numpy(float), 0, None))
        cols = [X]
        for c in CAT:
            cols.append(np.stack([(df[c] == v).to_numpy(float) for v in vocab[c]], 1))
        return np.hstack(cols)

    Xtr, Xte = enc(tr), enc(te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return (Xtr.astype(np.float32), tr.label.to_numpy(np.float32), tr.attack_cat.to_numpy(),
            Xte.astype(np.float32), te.label.to_numpy(np.float32), te.attack_cat.to_numpy())


def partition(cats, k, rng, alpha=None):
    """Split row indices across k clients. alpha=None gives IID; otherwise a
    Dirichlet(alpha) split per attack category (label skew, non-IID)."""
    n = len(cats)
    if alpha is None:
        return np.array_split(rng.permutation(n), k)
    parts = [[] for _ in range(k)]
    for c in np.unique(cats):
        idx = rng.permutation(np.where(cats == c)[0])
        p = rng.dirichlet([alpha] * k)
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for j, s in enumerate(np.split(idx, cuts)):
            parts[j].extend(s.tolist())
    return [np.array(sorted(p)) for p in parts]
