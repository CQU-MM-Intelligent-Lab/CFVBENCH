import re
import unicodedata
from typing import Set, List, Dict, Tuple
import math
import random

# Hardcoded defaults as requested
EASYOCR_CONF_THR = 0.3
OCR_JACCARD_THR = 0.85


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize('NFKC', s)
    s = s.lower()
    s = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens_from_text(s: str) -> Set[str]:
    s = normalize_text(s)
    if not s:
        return set()
    toks = [t for t in s.split() if len(t) >= 1]
    return set(toks)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = a & b
    uni = a | b
    return float(len(inter)) / float(len(uni))


# ------------------ MinHash + Simple LSH ------------------
def _hashfuncs(num_perm: int, max_shingle_id: int = 2**32 - 1) -> List[Tuple[int, int, int]]:
    # returns list of (a,b,prime) linear hash params
    res = []
    # use a fixed prime larger than max_shingle_id
    prime = 4294967311
    rand = random.Random(42)
    for _ in range(num_perm):
        a = rand.randint(1, prime - 1)
        b = rand.randint(0, prime - 1)
        res.append((a, b, prime))
    return res


def minhash_signature(token_set: Set[str], num_perm: int = 64) -> List[int]:
    # Map tokens to ints deterministically via hash
    if not token_set:
        return [2**63 - 1] * num_perm
    max_shingle_id = 2**32 - 1
    funcs = _hashfuncs(num_perm, max_shingle_id)
    sig = [2**63 - 1] * num_perm
    for tok in token_set:
        sh = (hash(tok) & 0xffffffff)
        for i, (a, b, p) in enumerate(funcs):
            val = (a * sh + b) % p
            if val < sig[i]:
                sig[i] = val
    return sig


def _lsh_bands(signature: List[int], band_size: int) -> List[Tuple[int, ...]]:
    # partition signature into bands; return list of tuples for each band
    if band_size <= 0:
        raise ValueError('band_size must be > 0')
    bands = []
    for i in range(0, len(signature), band_size):
        bands.append(tuple(signature[i:i+band_size]))
    return bands


def dedupe_texts_preserve_order_lsh(texts: List[str], thresh: float = OCR_JACCARD_THR, num_perm: int = 64, band_size: int = 8) -> List[str]:
    """
    Dedupe with MinHash+LSH acceleration. Returns kept texts in original order.
    - num_perm: number of hash permutations for MinHash (signature length)
    - band_size: signature band size for LSH (num_perm must be divisible by band_size ideally)
    """
    if not texts:
        return []
    token_sets = [tokens_from_text(t) for t in texts]
    sigs = [minhash_signature(ts, num_perm=num_perm) for ts in token_sets]
    # build LSH buckets
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    for idx, sig in enumerate(sigs):
        bands = _lsh_bands(sig, band_size)
        for b_idx, b in enumerate(bands):
            key = (b_idx, b)
            buckets.setdefault(key, []).append(idx)

    kept = []
    kept_token_sets: List[Set[str]] = []
    for i, t in enumerate(texts):
        ts = token_sets[i]
        if not ts:
            continue
        is_dup = False
        # Candidate indices from LSH: union of bucket members for this signature's bands
        cand_set = set()
        for b_idx, b in enumerate(_lsh_bands(sigs[i], band_size)):
            cand_set.update(buckets.get((b_idx, b), []))
        # check real Jaccard against candidates (only earlier kept ones)
        for k_idx, ks in enumerate(kept_token_sets):
            if k_idx in cand_set or True:
                if jaccard(ts, ks) >= thresh:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(t)
            kept_token_sets.append(ts)
    return kept


def dedupe_texts_preserve_order(texts, thresh: float = OCR_JACCARD_THR):
    # fallback simple method retained for small lists
    kept = []
    kept_token_sets = []
    for t in texts:
        ntoks = tokens_from_text(t)
        if not ntoks:
            continue
        is_dup = False
        for ks in kept_token_sets:
            if jaccard(ntoks, ks) >= thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(t)
            kept_token_sets.append(ntoks)
    return kept

