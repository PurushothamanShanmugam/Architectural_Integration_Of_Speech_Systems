import numpy as np
import torch

def generate_markov_labels(num_frames: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    lang_seq = np.zeros(num_frames, dtype=np.int64)
    lang_seq[0] = 1
    for i in range(1, num_frames):
        r = np.random.rand()
        if lang_seq[i - 1] == 1:
            lang_seq[i] = 1 if r < 0.97 else 0
        else:
            lang_seq[i] = 0 if r < 0.95 else 1
    return torch.tensor(lang_seq, dtype=torch.long)
