import numpy as np
from LanguageModel import MiniLanguageModel
from softmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss

np.random.seed(0)

B, T = 4, 8
V = 50
D = 16
H = 32

token_ids = np.random.randint(0, V, size=(B,T), dtype=np.int64)

# LM 타깃: 한 칸 shift (다음 토큰 맞추기)
targets = np.roll(token_ids, shift=-1, axis=1)
    
mask = np.ones((B, T), dtype=bool)
mask[:, -1] = False

model = MiniLanguageModel(vocab_size=V, emb_dim=D, hidden_dim=H, seed=42)
loss_fn = SoftmaxCrossEntropyLoss() 

logits = model.forward(token_ids)
loss = model.loss(token_ids, targets, loss_fn, mask=mask)

print("token_ids shape:", token_ids.shape)   # (B,T)
print("logits shape:", logits.shape)         # (B,T,V)
print("loss:", loss)