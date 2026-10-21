import numpy as np
from securefl.data import partition
from securefl.model import MLP, local_train
from securefl.fl import topk, upload_bytes, metrics, epsilon


def test_partition_disjoint_and_complete():
    cats = np.repeat(np.array(list("abcde")), 200)
    for alpha in [None, 0.5, 0.1]:
        parts = partition(cats, 10, np.random.default_rng(0), alpha)
        allidx = np.concatenate(parts)
        assert len(allidx) == len(cats) and len(np.unique(allidx)) == len(cats)


def test_per_example_clipping_bounds_norm():
    rng = np.random.default_rng(0); X = rng.normal(size=(64, 7)).astype(np.float32); y = (rng.random(64) > .5).astype(np.float32)
    m = MLP(7, 5); w = m.init(rng)
    for i in range(8):  # each single-example gradient must have norm <= clip
        g = m.grad(w, X[i:i + 1] * 50, y[i:i + 1], clip=1.0)
        assert np.linalg.norm(g) <= 1.0 + 1e-4


def test_unclipped_gradient_matches_finite_difference():
    rng = np.random.default_rng(1); X = rng.normal(size=(5, 3)).astype(np.float64); y = np.array([1, 0, 1, 1, 0.])
    m = MLP(3, 4); w = m.init(rng).astype(np.float64)
    loss = lambda w: -np.sum(y * np.log(m.predict(w, X)) + (1 - y) * np.log(1 - m.predict(w, X)))
    g = m.grad(w, X, y); e = np.zeros_like(w); e[2] = 1e-6
    assert abs((loss(w + e) - loss(w - e)) / 2e-6 - g[2]) < 1e-3


def test_topk_and_bytes():
    u = np.arange(100, dtype=np.float32) - 50
    idx, val = topk(u, 0.1); assert len(idx) == 10 and set(np.abs(val)) >= {50.0}
    assert upload_bytes(1000, None) == 4000 and upload_bytes(1000, 0.1) == 600


def test_metrics_and_epsilon_monotone():
    r = metrics(np.array([0, 0, 1, 1]), np.array([.1, .6, .7, .9]))
    assert r["fpr"] == .5 and r["recall"] == 1 and abs(r["auc"] - 1) < 1e-9
    assert epsilon(10000, 256, 1000, 2.0) < epsilon(10000, 256, 1000, 1.0) < epsilon(10000, 256, 4000, 1.0)


def test_training_reduces_loss():
    rng = np.random.default_rng(2); X = rng.normal(size=(2000, 6)).astype(np.float32); y = (X[:, 0] > 0).astype(np.float32)
    m = MLP(6, 8); w = m.init(rng); w2, _ = local_train(m, w, X, y, rng, epochs=3, lr=0.2, batch=64)
    acc = lambda w: ((m.predict(w, X) > .5) == y).mean()
    assert acc(w2) > max(acc(w), 0.9)
