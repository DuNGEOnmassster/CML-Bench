"""Connected-component split assignment across works, editions and adaptations."""

import hashlib
from collections import defaultdict


def assign(records, seed="cml-literary-v1"):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    tokens = {}
    for record in records:
        ts = [
            "work:" + record["work_id"],
            "family:" + record["story_family_id"],
            "text:" + record["normalized_sha256"],
        ]
        ts += ["family:" + x for x in record.get("related_family_ids", [])]
        tokens[record["record_id"]] = ts
        for t in ts:
            union(ts[0], t)
    groups = defaultdict(set)
    for ts in tokens.values():
        groups[find(ts[0])].update(ts)
    result = {}
    for record in records:
        group = groups[find(tokens[record["record_id"]][0])]
        # Family identity rather than edition list controls split stability.
        families = sorted(t for t in group if t.startswith("family:"))
        group_id = hashlib.sha256("\n".join(families).encode()).hexdigest()[:20]
        bucket = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) % 100
        result[record["record_id"]] = {
            "split": "train" if bucket < 80 else "validation" if bucket < 90 else "test",
            "split_group": group_id,
        }
    return result
