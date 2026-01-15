import numpy as np
from embedding import Embedding
from rnn import VanillaRNN
from linear import Linear

class MiniLanguageModel:
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim:int, seed:int | None= None):
        self.embed = Embedding(vocab_size, emb_dim, seed=seed)
        self.rnn = VanillaRNN(emb_dim, hidden_dim, seed=None if seed is None else seed + 1)
        self.proj = Linear(hidden_dim, vocab_size, seed=None if seed is None else seed+2)

    def forward(self, token_ids:  np.ndarray):
        """
        token_ids: (B, T)
        return logits: (B, T, V)
        """
        x = self.embed.forward(token_ids)
        h0 = self.rnn.forward(x, None)
        logits = self.proj.forward(h0)
        return logits
    
    def loss(self, token_ids: np.ndarray, targets: np.ndarray, loss_fn, mask: np.ndarray | None = None) -> float:
        logits = self.forward(token_ids)
        return loss_fn.forward(logits, targets, mask)