#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Independent test oracle for tests/omegese45/core/run-core.sh (aien-train #45, pipeline
# lane). Python stdlib only; it is a referee for the .om code, never part of the pipeline.
#   oracle.py mask VOCAB.tsv EPISODES.oe45 K        expected `M e t target mask` lines
#   oracle.py policy EPISODES.oe45                  reference-policy check (rc 0 / 1)
#   oracle.py prefixes EPISODES.oe45 K OUT EXPECT   decode prefixes and expected next messages
import sys

HDR = "oe45 v1"


def episodes(path):
    eps, cur = [], None
    with open(path, encoding="ascii") as f:
        for line in f.read().split("\n")[:-1]:
            if line == HDR:
                cur = []
                eps.append(cur)
            else:
                cur.append(line.split(" "))
    return eps


def ids_of(words, vocab):
    out = []
    for w in words:
        if w.startswith("0x"):
            out.append(vocab["0x"])
            out.extend(vocab[c] for c in w[2:])
        else:
            out.append(vocab[w])
    return out


def cmd_mask(vocab_path, ep_path, k):
    vocab = {}
    with open(vocab_path, encoding="ascii") as f:
        for line in f:
            i, tok = line.split("\t")[:2]
            vocab[tok] = int(i)
    for e, msgs in enumerate(episodes(ep_path)[: int(k)]):
        t = 0
        for m in msgs:
            ap = 1 if m[0] in ("act", "rep") else 0
            for i in ids_of(m, vocab):
                print(f"M {e} {t} {i} {ap}")
                t += 1
    return 0


def check_policy(msgs):
    g = msgs[0]
    if g[0] != "goal":
        return "no goal"
    kind, dims = g[1], g[2:8]
    seeds = [g[9]] if kind == "mm" else [g[9], g[11]]
    vals = [int(dims[i], 16) for i in (1, 3, 5)]
    rest = msgs[1:]
    if any(v < 1 or v > 256 for v in vals):
        want = [["act", "refuse", "bound", "end"], ["rep", "refused", "bound", "end"]]
        return None if rest == want else "bound"
    pos = 0
    for s in seeds:
        tries = 0
        while True:
            a = ["act", "job", "mm"] + dims + ["ulong", s, "end"]
            if rest[pos] != a:
                return "act"
            r = rest[pos + 1]
            pos += 2
            if r == ["res", "ok", "end"]:
                break
            if r == ["res", "fail", "transport", "end"] and tries == 0:
                tries = 1
                continue
            want = [["rep", r[1], r[2], "end"]]
            return None if rest[pos:] == want else "stop"
    want = [["rep", "ok"] + g[1:]]
    return None if rest[pos:] == want else "report"


def cmd_policy(ep_path):
    bad = 0
    eps = episodes(ep_path)
    for e, msgs in enumerate(eps):
        why = check_policy(msgs)
        if why:
            bad += 1
            print(f"POLICY_BAD {e} {why}")
    print(f"POLICY {len(eps)} {bad}")
    return 1 if bad else 0


def cmd_prefixes(ep_path, k, out_path, exp_path):
    with open(out_path, "w", encoding="ascii") as out, open(exp_path, "w", encoding="ascii") as exp:
        for msgs in episodes(ep_path)[: int(k)]:
            for i in range(1, len(msgs)):
                prev = msgs[i - 1]
                # Every A/P position: after G or R, and the P that follows `act refuse` (no R).
                if msgs[i][0] in ("act", "rep") and (prev[0] in ("goal", "res") or prev[:2] == ["act", "refuse"]):
                    out.write(HDR + "\n" + "".join(" ".join(m) + "\n" for m in msgs[:i]))
                    exp.write(" ".join(msgs[i]) + "\n")
    return 0


def main(argv):
    if len(argv) == 5 and argv[1] == "mask":
        return cmd_mask(argv[2], argv[3], argv[4])
    if len(argv) == 3 and argv[1] == "policy":
        return cmd_policy(argv[2])
    if len(argv) == 6 and argv[1] == "prefixes":
        return cmd_prefixes(argv[2], argv[3], argv[4], argv[5])
    print("usage: oracle.py mask VOCAB EPS K | policy EPS | prefixes EPS K OUT EXPECT", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
