import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, seed: int | None=None):
        rng = np.random.default_rng(seed)
        self.H = hidden_dim
        self.W_xh = (rng.standard_normal((hidden_dim, input_dim)) * 0.02).astype(np.float32)
        self.W_hh = (rng.standard_normal((hidden_dim, hidden_dim)) * 0.02).astype(np.float32) 
        self.b_h = np.zeros((hidden_dim,), dtype=np.float32)

    def forward(self, x: np.ndarray, h0: np.ndarray):
        """
        x:               (B,T,D)
        h0:            (B, H) or None -> zeros
        return:      (B,T,G)
        """
        B,T,D = x.shape
        if h0 is None:
            h = np.zeros((B, self.H), dtype=np.float32)
        else:
            assert h0.shape == (B, self.H)
            h = h0.astype(np.float32)
        hs= []
        for t in range(T):
            x_t = x[:, t, :]
            h = np.tanh(x_t @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
            hs.append(h.astype(np.float32))
        return np.stack(hs, axis=1)
    
    def backward():
        pass
    def zero_grad():
        pass
    