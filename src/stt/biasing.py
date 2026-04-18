from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

SYLLABUS_TEXT = """
speech recognition acoustic model language model hidden Markov model stochastic process cepstrum
mel frequency cepstral coefficients MFCC feature extraction dynamic time warping DTW edit distance
phoneme phonetics prosody fundamental frequency pitch energy duration formant spectrogram short-time
Fourier transform STFT windowing Hamming window filterbank Gaussian mixture model GMM deep neural
network DNN recurrent neural network LSTM attention mechanism transformer encoder decoder CTC beam
search decoding word error rate WER speaker diarization speaker recognition x-vector ECAPA-TDNN
"""

@dataclass
class SimpleNGramLM:
    vocab: List[str]
    counts: Dict[Tuple[str, ...], Counter]

def build_ngram_lm(text: str, n: int = 3) -> SimpleNGramLM:
    tokens = text.lower().split()
    counts = defaultdict(Counter)
    for k in range(1, n + 1):
        for i in range(len(tokens) - k + 1):
            hist = tuple(tokens[i:i + k - 1])
            nxt = tokens[i + k - 1]
            counts[hist][nxt] += 1
    return SimpleNGramLM(vocab=sorted(set(tokens)), counts=dict(counts))

def get_initial_prompt() -> str:
    return (
        "Lecture on speech processing. Technical terms: "
        "stochastic cepstrum MFCC acoustic model language model DTW "
        "CTC beam search WER VITS speaker embedding code-switching."
    )
