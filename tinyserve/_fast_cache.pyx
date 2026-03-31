# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-accelerated cache operations for tinyserve hot path."""

from libc.math cimport INFINITY


def lfru_select_evict(dict data, long clock):
    """Find the key with lowest freq/age score in LFRU data dict.

    Args:
        data: dict mapping tuple keys to [slot, freq, clock] lists
        clock: current LFRU clock value

    Returns:
        (best_key, slot) — the entry to evict
    """
    cdef double best_score = INFINITY
    cdef double score
    cdef long freq, last, age, slot
    cdef object best_key = None

    for key, entry in data.items():
        slot = entry[0]
        freq = entry[1]
        last = entry[2]
        age = clock - last + 1
        score = <double>freq / <double>age
        if score < best_score:
            best_score = score
            best_key = key

    return best_key, data[best_key][0]


def classify_hits_misses(list expert_ids_list, list slots_list):
    """Classify experts as hits or misses from slot lookup results.

    Args:
        expert_ids_list: list of int expert IDs
        slots_list: list of int slot indices (-1 = miss)

    Returns:
        (hits, misses) where hits is list of (index, slot) and misses is list of index
    """
    cdef list hits = []
    cdef list misses = []
    cdef int i, slot
    cdef int n = len(expert_ids_list)

    for i in range(n):
        slot = slots_list[i]
        if slot >= 0:
            hits.append((i, slot))
        else:
            misses.append(i)

    return hits, misses


def group_tokens_by_expert(list eid_list, int seq_len, int top_k):
    """Group token indices by expert ID for batched prefill.

    Args:
        eid_list: flattened list of expert IDs [seq_len][top_k] as nested list
        seq_len: number of tokens
        top_k: experts per token

    Returns:
        dict mapping expert_id to list of (token_idx, k_idx) tuples
    """
    cdef dict groups = {}
    cdef int tok, k, eid
    cdef list group

    for tok in range(seq_len):
        for k in range(top_k):
            eid = eid_list[tok][k]
            if eid in groups:
                group = groups[eid]
            else:
                group = []
                groups[eid] = group
            group.append((tok, k))

    return groups
