"""One-hidden-layer MLP held as a flat float32 vector, with optional
per-example clipping and Gaussian noise (DP-SGD, Abadi et al. 2016)."""
import numpy as np


class MLP:
    def __init__(self, d, h=64):
        self.d, self.h = d, h
        self.shapes = [(d, h), (h,), (h,), (1,)]
        self.size = sum(int(np.prod(s)) for s in self.shapes)

    def init(self, rng):
        w = np.zeros(self.size, np.float32)
        w[: self.d * self.h] = rng.normal(0, np.sqrt(2 / self.d), self.d * self.h)
        o = self.d * self.h + self.h
        w[o:o + self.h] = rng.normal(0, np.sqrt(1 / self.h), self.h)
        return w

    def unpack(self, w):
        out, o = [], 0
        for s in self.shapes:
            n = int(np.prod(s)); out.append(w[o:o + n].reshape(s)); o += n
        return out

    def predict(self, w, X):
        W1, b1, w2, b2 = self.unpack(w)
        z = np.maximum(X @ W1 + b1, 0) @ w2 + b2
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def grad(self, w, X, y, clip=None):
        """Sum of (optionally clipped) per-example gradients of BCE loss."""
        W1, b1, w2, b2 = self.unpack(w)
        a = X @ W1 + b1; hdn = np.maximum(a, 0)
        p = 1 / (1 + np.exp(-np.clip(hdn @ w2 + b2, -30, 30)))
        dz = p - y
        dh = (dz[:, None] * w2[None, :]) * (a > 0)
        if clip is not None:
            # per-example squared norm; W1 grad is an outer product x_i dh_i^T
            sq = (X * X).sum(1) * (dh * dh).sum(1) + (dh * dh).sum(1) \
                + dz * dz * ((hdn * hdn).sum(1) + 1)
            c = np.minimum(1.0, clip / (np.sqrt(sq) + 1e-12)).astype(np.float32)
            dz, dh = dz * c, dh * c[:, None]
        return np.concatenate([(X.T @ dh).ravel(), dh.sum(0), hdn.T @ dz, [dz.sum()]]).astype(np.float32)


def local_train(m, w, X, y, rng, epochs=1, lr=0.05, batch=256, clip=None, z=0.0):
    """Mini-batch SGD. With clip set, uses Poisson sampling at rate batch/n,
    per-example clipping and N(0, (z*clip)^2) noise: standard DP-SGD."""
    w = w.copy(); n = len(y)
    steps = max(1, int(round(epochs * n / batch)))
    if clip is None:
        for _ in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                b = perm[s:s + batch]
                w -= lr * m.grad(w, X[b], y[b]) / len(b)
        return w, 0
    q = batch / n
    for _ in range(steps):
        b = np.where(rng.random(n) < q)[0]
        g = m.grad(w, X[b], y[b], clip=clip) if len(b) else np.zeros_like(w)
        g = g + rng.normal(0, z * clip, w.size).astype(np.float32)
        w -= lr * g / batch
    return w, steps
