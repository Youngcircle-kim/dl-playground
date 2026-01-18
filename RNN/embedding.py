import numpy as np

class Embedding:
    def __init__(self, vocab_size: int, emb_dim: int, seed: int | None=None):
        rng = np.random.default_rng(seed)
        self.E = (rng.standard_normal((vocab_size, emb_dim)) * 0.02).astype(np.float32)
        self.token_ids = None
        self.dE = np.zeros_like(self.E)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        token_ids: (B,T) int
        return:       (B, T, D)
        """
        self.token_ids = token_ids
        return self.E[token_ids]

    def backward(self, dx: np.ndarray) -> None:
        self.dE.fill(0.0)
        ids = self.token_ids.reshape(-1)
        grad = dx.reshape(-1, dx.shape[-1])
        np.add.at(self.dE, ids, grad)

    def step(self, lr: float):
        self.dE -= lr * self.dE