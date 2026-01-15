import numpy as np

class Linear:
    def __init__(self, in_dim: int, out_dim:int, seed: int):
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((out_dim,in_dim)) * 0.02).astype(np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)
    def forward(self, x: np.ndarray):
        """
        x:          (B, T, H)
        return: (B,T,V)
        """
        return x @ self.W.T + self.b
    
    def backward():
        pass