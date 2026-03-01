import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}


def normalize_phrase(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def phrase_letters(phrase: str) -> str:
    return normalize_phrase(phrase).replace(" ", "")


def extract_phrase_from_filename(filename: str) -> Optional[str]:
    # Example: AB，help me，1组-xxxx.zip -> HELP ME
    m = re.search(r"AB[,，]\s*(.*?)[,，]", filename, flags=re.IGNORECASE)
    if not m:
        return None
    candidate = normalize_phrase(m.group(1))
    if not candidate:
        return None
    return candidate


def load_lexicon(lexicon_file=None) -> List[str]:
    if lexicon_file is None:
        return []
    path = Path(lexicon_file)
    if not path.exists():
        return []
    lines = [normalize_phrase(x.strip()) for x in path.read_text(encoding="utf-8").splitlines()]
    lines = [x for x in lines if x]
    return sorted(set(lines))


def infer_lexicon_from_filenames(filenames: Iterable[str]) -> List[str]:
    phrases = []
    for name in filenames:
        p = extract_phrase_from_filename(name)
        if p:
            phrases.append(p)
    return sorted(set(phrases))


@dataclass
class DecodeResult:
    decoded_phrase: str
    decoded_letters: str
    score: float


def _raw_score(log_probs: np.ndarray) -> float:
    # Average best per-step log-prob.
    return float(np.max(log_probs, axis=1).mean())


def _score_phrase_dp(
    log_probs: np.ndarray,
    letters: str,
    ins_penalty: float = -2.5,
    del_penalty: float = -2.0,
):
    """
    Align T segment probabilities to L target letters using DP.
    Match: consume one segment + one letter.
    Delete-segment: consume one segment only.
    Insert-letter: consume one letter only.
    """
    T = int(log_probs.shape[0])
    L = len(letters)
    if T == 0:
        return -1e9
    if L == 0:
        return _raw_score(log_probs)

    neg_inf = -1e18
    dp = np.full((T + 1, L + 1), neg_inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for t in range(1, T + 1):
        dp[t, 0] = dp[t - 1, 0] + del_penalty
    for l in range(1, L + 1):
        dp[0, l] = dp[0, l - 1] + ins_penalty

    for t in range(1, T + 1):
        for l in range(1, L + 1):
            ch = letters[l - 1]
            idx = CHAR_TO_IDX.get(ch)
            if idx is None:
                continue

            match = dp[t - 1, l - 1] + log_probs[t - 1, idx]
            del_seg = dp[t - 1, l] + del_penalty
            ins_char = dp[t, l - 1] + ins_penalty
            dp[t, l] = max(match, del_seg, ins_char)

    norm = float(max(T, L))
    return float(dp[T, L] / norm)


def decode_with_lexicon(
    prob_matrix: np.ndarray,
    raw_letters: str,
    lexicon: List[str],
    min_gain: float = -0.15,
    force_lexicon: bool = False,
) -> DecodeResult:
    """
    Compare unconstrained decoding against lexicon-constrained decoding.
    min_gain is permissive by default to stabilize outputs with small score loss.
    """
    if prob_matrix.size == 0:
        return DecodeResult(decoded_phrase="", decoded_letters="", score=-1e9)

    clipped = np.clip(prob_matrix, 1e-8, 1.0)
    log_probs = np.log(clipped)
    baseline = _raw_score(log_probs)

    best_phrase = ""
    best_letters = ""
    best_score = -1e18

    for phrase in lexicon:
        letters = phrase_letters(phrase)
        if not letters:
            continue
        score = _score_phrase_dp(log_probs, letters)
        if score > best_score:
            best_score = score
            best_phrase = phrase
            best_letters = letters

    if best_phrase and (force_lexicon or best_score >= baseline + min_gain):
        return DecodeResult(decoded_phrase=best_phrase, decoded_letters=best_letters, score=best_score)
    return DecodeResult(decoded_phrase="", decoded_letters=raw_letters, score=baseline)
