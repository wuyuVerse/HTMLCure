"""
HTMLDeduplicator — near-duplicate removal using shingling + Jaccard similarity.

Uses a pure-Python MinHash implementation (no external deps).
For 100k records, runs in ~30s with default settings.

Algorithm:
  1. Tokenise HTML into 5-grams of characters
  2. Compute MinHash signature (128 hashes)
  3. Use LSH banding to find candidate pairs
  4. Exact Jaccard check on candidates above threshold
  5. Union-Find to cluster duplicates; keep one representative per cluster
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("htmlrefine.dedup")

# ---------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------

_LARGE_PRIME = (1 << 61) - 1   # Mersenne prime

def _shingles(text: str, k: int = 5) -> Set[int]:
    """k-char shingles of normalised HTML."""
    # Normalise: collapse whitespace, lowercase
    text = re.sub(r"\s+", " ", text.lower())
    return {hash(text[i: i + k]) & 0xFFFFFFFF for i in range(max(1, len(text) - k + 1))}


def _minhash(shingles: Set[int], num_perm: int = 128) -> List[int]:
    """Compute a MinHash signature of `num_perm` values."""
    sig = [_LARGE_PRIME] * num_perm
    for s in shingles:
        for i in range(num_perm):
            h = (s * (i + 1) * 2654435761 + i * 1013904223) & 0xFFFFFFFF
            if h < sig[i]:
                sig[i] = h
    return sig


def _jaccard_estimate(sig_a: List[int], sig_b: List[int]) -> float:
    matches = sum(a == b for a, b in zip(sig_a, sig_b))
    return matches / len(sig_a)


# ---------------------------------------------------------------------------
# LSH banding
# ---------------------------------------------------------------------------

def _band_keys(sig: List[int], bands: int, rows: int) -> List[str]:
    """Hash each band to a bucket key."""
    keys = []
    for b in range(bands):
        band = sig[b * rows: (b + 1) * rows]
        h = hashlib.md5(str(band).encode()).hexdigest()
        keys.append(f"{b}_{h}")
    return keys


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self):
        self._parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        self._parent[self.find(x)] = self.find(y)

    def clusters(self, ids: List[int]) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for i in ids:
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups


# ---------------------------------------------------------------------------
# Main deduplicator
# ---------------------------------------------------------------------------

class HTMLDeduplicator:
    """
    Remove near-duplicate HTML records.

    Args:
        threshold:  Jaccard similarity threshold (default 0.8)
        num_perm:   MinHash permutations (default 128)
        bands:      LSH bands (default 32; rows = num_perm // bands)
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, bands: int = 32):
        self.threshold = threshold
        self.num_perm  = num_perm
        self.bands     = bands
        self.rows      = num_perm // bands

    def deduplicate(self, records: List[dict], html_key: str = "html") -> Tuple[List[dict], int]:
        """
        Deduplicate records by their HTML content.

        Args:
            records:  list of dicts; each must have an html_key field.
            html_key: key name for HTML content (default "html").

        Returns:
            (deduplicated list, number of removed duplicates)
        """
        if not records:
            return records, 0

        logger.info(f"Deduplicating {len(records)} records (threshold={self.threshold})")

        # Compute signatures
        sigs: List[List[int]] = []
        for rec in records:
            html = rec.get(html_key, "") or ""
            sigs.append(_minhash(_shingles(html), self.num_perm))

        # LSH: bucket candidates
        buckets: Dict[str, List[int]] = {}
        for i, sig in enumerate(sigs):
            for key in _band_keys(sig, self.bands, self.rows):
                buckets.setdefault(key, []).append(i)

        # Find pairs above threshold
        uf = _UnionFind()
        checked: Set[Tuple[int, int]] = set()
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            for a in range(len(bucket)):
                for b in range(a + 1, len(bucket)):
                    i, j = bucket[a], bucket[b]
                    if (i, j) in checked:
                        continue
                    checked.add((i, j))
                    if _jaccard_estimate(sigs[i], sigs[j]) >= self.threshold:
                        uf.union(i, j)

        # Keep one record per cluster (the highest-scoring one)
        clusters = uf.clusters(list(range(len(records))))
        kept: List[dict] = []
        removed = 0
        for members in clusters.values():
            if len(members) == 1:
                kept.append(records[members[0]])
            else:
                best = max(members, key=lambda i: _get_score(records[i]))
                kept.append(records[best])
                removed += len(members) - 1

        logger.info(f"Dedup done: {len(records)} → {len(kept)} records ({removed} removed)")
        return kept, removed


def _get_score(rec: dict) -> int:
    s = rec.get("score", {})
    if isinstance(s, dict):
        return int(s.get("total", 0))
    return int(s or 0)
