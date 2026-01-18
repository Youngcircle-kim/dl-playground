import numpy as np
from LanguageModel import MiniLanguageModel
from softmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss

if __name__ == "__main__":
    np.random.seed(0)

    B, T = 4, 8
    V = 50
    D = 16
    H = 32

    token_ids = np.random.randint(0, V, size=(B, T), dtype=np.int64)
    targets = np.roll(token_ids, shift=-1, axis=1)

    mask = np.ones((B, T), dtype=bool)
    mask[:, -1] = False

    model = MiniLanguageModel(vocab_size=V, emb_dim=D, hidden_dim=H, seed=42)
    loss_fn = SoftmaxCrossEntropyLoss()

    print("random baseline ~ log(V) =", np.log(V))

    lr = 0.5
    for step in range(1, 201):
        loss = model.train_step(token_ids, targets, loss_fn, lr=lr, mask=mask)
        if step % 20 == 0:
            print(f"step {step:3d} | loss {loss:.4f}")