import numpy as np
from embedding import Embedding
from rnn import VanillaRNN
from linear import Linear

class MiniLanguageModel:
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, seed: int | None = None):
        self.embed = Embedding(vocab_size, emb_dim, seed=seed)
        self.rnn   = VanillaRNN(emb_dim, hidden_dim, seed=None if seed is None else seed + 1)
        self.proj  = Linear(hidden_dim, vocab_size, seed=None if seed is None else seed + 2)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        x = self.embed.forward(token_ids)     # (B,T,D)
        h = self.rnn.forward(x)               # (B,T,H)
        logits = self.proj.forward(h)         # (B,T,V)
        return logits

    def train_step(self, token_ids, targets, loss_fn, lr: float, mask=None) -> float:
        logits = self.forward(token_ids)
        loss = loss_fn.forward(logits, targets, mask)

        dlogits = loss_fn.backward()          # (B,T,V)
        dh = self.proj.backward(dlogits)      # (B,T,H)
        dx = self.rnn.backward(dh)            # (B,T,D)
        self.embed.backward(dx)               # dE

        # SGD update
        self.proj.step(lr)
        self.rnn.step(lr)
        self.embed.step(lr)
        return loss