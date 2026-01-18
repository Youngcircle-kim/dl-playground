import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, seed: int | None=None):
        rng = np.random.default_rng(seed)
        self.H = hidden_dim
        self.W_xh = (rng.standard_normal((hidden_dim, input_dim)) * 0.02).astype(np.float32)
        self.W_hh = (rng.standard_normal((hidden_dim, hidden_dim)) * 0.02).astype(np.float32)
        self.b_h  = np.zeros((hidden_dim,), dtype=np.float32)

        # cache
        self.x = None
        self.hs = None
        self.h0 = None

        # grads
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h  = np.zeros_like(self.b_h)


    def forward(self, x: np.ndarray, h0 = None) -> np.ndarray:
        """
        x:               (B,T,D)
        h0:            (B, H) or None -> zeros
        return:      (B,T,G)
        """
        B,T,D = x.shape
        self.x = x
        self.h0 = np.zeros((B, self.H), dtype=np.float32) if h0 is None else h0.astype(np.float32)

        hs = np.zeros((B,T, self.H), dtype=np.float32)
        h_prev = self.h0
        for t in range(T):
            a = x[:, t, :] @ self.W_xh.T + h_prev @ self.W_hh.T + self.b_h
            h = np.tanh(a).astype(np.float32)
            hs[:, t, :] = h
            h_prev = h
        self.hs = hs
        return hs
    
    def backward(self, dhs: np.ndarray):
        x = self.x
        hs = self.hs
        B,T,D = x.shape
        H = self.H

        self.dW_xh.fill(0.0)
        self.dW_hh.fill(0.0)
        self.db_h.fill(0.0)

        dx = np.zeros_like(x, dtype=np.float32)
        dh_next = np.zeros((B,H), dtype=np.float32)

        for t in reversed(range(T)):
            h_t = hs[:, t, :]
            h_prev = self.h0 if t == 0 else hs[:, t-1, :]

            dh = dhs[:, t, :] + dh_next                       # (B,H)
            da = dh * (1.0 - h_t**2)                           # tanh'(a) = 1-h^2

            self.db_h += da.sum(axis=0)
            self.dW_xh += da.T @ x[:, t, :]
            self.dW_hh += da.T @ h_prev

            dx[:, t, :] = da @ self.W_xh
            dh_next = da @ self.W_hh

        return dx
    
    def step(self, lr: float):
        self.W_xh -= lr * self.dW_xh
        self.W_hh -= lr * self.dW_hh
        self.b_h  -= lr * self.db_h

    