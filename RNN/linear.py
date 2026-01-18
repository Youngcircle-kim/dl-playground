import numpy as np

class Linear:
    def __init__(self, in_dim: int, out_dim:int, seed: int):
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((out_dim,in_dim)) * 0.02).astype(np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray):
        """
        x:          (B, T, H)
        return: (B,T,V)
        """
        self.x = x
        return x @ self.W.T + self.b
    
    def backward(self, dlogits: np.ndarray):
        # dlogits: (B,T,V)
        x = self.x
        B, T, H = x.shape
        V = dlogits.shape[-1]

        x2 = x.reshape(B*T, H)
        dl2 = dlogits.reshape(B*T, V)

        self.dW = (dl2.T @ x2).astype(np.float32)
        self.db = dl2.sum(axis=0).astype(np.float32)
        dx2 = dl2 @ self.W
        return dx2.reshape(B, T, H).astype(np.float32)
    
    def step(self, lr:float) -> None:
        self.W -= lr * self.dW
        self.b -= lr * self.db