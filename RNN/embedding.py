import numpy as np

class Embedding:
    def __init__(self, vocab_size: int, emb_dim: int, seed: int | None=None):
        rng = np.random.default_rng(seed)
        self.E = (rng.standard_normal((vocab_size, emb_dim)) * 0.02).astype(np.float32)

    def forward(self, token_ids: np.ndarray):
        """
        token_ids: (B,T) int
        return:       (B, T, D)
        """
        return self.E[token_ids]
    
    def backward():
        pass
    def zero_grad():
        pass