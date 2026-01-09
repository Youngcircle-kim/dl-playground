import numpy as np

class SoftmaxCrossEntropyLoss:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.probs = None
        self.targets = None
        self.mask = None
        self.den = None

    def forward(self, logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None) -> float:
        """
        B: Batch size, T: Time steps, V: Vocablary size
        logits:  (B, T, V)          / 모델이 예측한 "다음 토큰 점수(Raw score)"
        targets: (B, T) int       / Next token의 "정수 ID"
        mask:    (B, T) bool    / 이 위치의 Loss를 계산할지 말지 결정하는 스위치
        return:  scalar loss    / 전체 배치, 시간에 대한 평균 손실 값
        """
        B,T,V = logits.shape
        assert targets.shape == (B, T)

        x = logits - logits.max(axis=-1, keepdims=True)                                 # (B, T, V)
        expx = np.exp(x)
        probs = expx / (expx.sum(axis= -1, keepdims=True) + self.eps)

        flat_probs = probs.reshape(-1, V)                                                          #  (B*T, V)
        flat_tgt = targets.reshape(-1)                                                                 #  (B*T, )
        p_true = flat_probs[np.arange(B*T), flat_tgt]                                      #  (B*T, )

        nll = -np.log(p_true + self.eps)                                                               #  (B*T, )

        if mask is not None:
            assert mask.shape == (B, T)
            flat_mask = mask.reshape(-1).astype(np.float32)
            nll = nll * flat_mask
            den = float(flat_mask.sum()) + self.eps
        else:
            den = float(B * T)
        loss = float(nll.sum() / den)

        # cache (for Backpropagation)
        self.probs = probs
        self.targets = targets
        self.mask = mask
        self.den = den
        return loss

    def backward(self) -> np.ndarray:
        """
        return dlogits: (B, T, V)
        """
        probs = self.probs
        targets = self.targets
        assert probs is not None and targets is not None and self.den is not None

        B, T, V = probs.shape
        dlogits = probs.copy().reshape(-1, V)                                                   # (B*T, V)
        flat_tgt = targets.reshape(-1)
        dlogits[np.arange(B*T), flat_tgt] -= 1.0

        dlogits = dlogits.reshape(B,T,V)
        
        if self.mask is not None:
            dlogits *= self.mask[..., None].astype(np.float32)
        
        dlogits /= self.den
        return dlogits