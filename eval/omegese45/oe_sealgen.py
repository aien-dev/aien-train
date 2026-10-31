#!/usr/bin/env python3
"""Independent sealed-set generator for aien-train #45 (eval lane).

Usage: oe_sealgen.py SEEDFILE OUTFILE

SEEDFILE holds one line: 64 lowercase hex characters (the eval lineage seed).
OUTFILE receives the sealed bundle: for each episode a line
`@ <id> <stratum>` followed by the OE45 v1 episode lines (`oe45 v1`, messages).

Written from interface v1.1 sections 2, 3 and 7 only. It does not import or
read the pipeline generator, the translator or the contract lane's code.
Randomness is a SHA-256 counter-mode stream over the seed (no dependence on
the Python `random` module's version-specific algorithms). Strata and the
outcome distribution are declared in docs/omegese45/EVAL-CONTRACT.md §3.
"""
import hashlib
import sys

if __package__ in (None, ""):
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from oe45lib import (ACT_REFUSE, EPISODE_HEADER, RCODES, RUNTIME_CLASS, admitted,
                     fmt_act, fmt_goal, fmt_rep, fmt_rep_ok, fmt_res)

GENERATOR_ID = "oe_sealgen v1"


class Stream:
    """Deterministic integer stream: SHA-256(seed || label || counter)."""

    def __init__(self, seed_bytes, label):
        self.key = seed_bytes + b"|" + label.encode()
        self.ctr = 0
        self.buf = b""

    def _bytes(self, n):
        while len(self.buf) < n:
            self.buf += hashlib.sha256(self.key + self.ctr.to_bytes(8, "big")).digest()
            self.ctr += 1
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def below(self, n):
        """Uniform integer in [0, n) by rejection sampling over 32-bit draws."""
        assert 0 < n <= 1 << 32
        lim = (1 << 32) - ((1 << 32) % n)
        while True:
            v = int.from_bytes(self._bytes(4), "big")
            if v < lim:
                return v % n

    def rng(self, lo, hi):
        return lo + self.below(hi - lo + 1)

    def pick(self, seq):
        return seq[self.below(len(seq))]


NONT = [c for c in RCODES if c != "transport"]  # 7 non-transport C' codes


def res_for(code):
    """Runtime R for a C' code, with its aienet0-run class."""
    return ("ok", None) if code == "ok" else (RUNTIME_CLASS[code], code)


def build_episode(kind, dims, seeds, script):
    """Write one episode from a per-job outcome script.

    script: list (one entry per job) of attempt lists; each attempt is an R as
    (cls, code). Attempt patterns follow interface §3.2: [ok], [fail transport,
    ok], [fail transport, fail transport], [X], [fail transport, X].
    Returns the list of message lines (without the header).
    """
    lines = [fmt_goal(kind, dims, seeds)]
    if any(not admitted(d) for d in dims):
        return lines + [ACT_REFUSE, fmt_rep("refused", "bound")]
    for j, attempts in enumerate(script):
        for a, (cls, code) in enumerate(attempts):
            lines.append(fmt_act(dims, seeds[j]))
            lines.append(fmt_res(cls, code))
            if cls == "ok":
                break
            if cls == "fail" and code == "transport" and a == 0:
                continue  # first transport failure on this job: retry once
            return lines + [fmt_rep(cls, code)]
    return lines + [fmt_rep_ok(kind, dims, seeds)]


def check_script(kind, script):
    njobs = 1 if kind == "mm" else 2
    assert 1 <= len(script) <= njobs
    for j, att in enumerate(script):
        assert 1 <= len(att) <= 2
        if len(att) == 2:
            assert att[0] == ("fail", "transport")
        last = att[-1]
        stopped = last != ("ok", None)
        if stopped:
            assert j == len(script) - 1, "no job after a stop"
    if len(script) < njobs:
        assert script[-1][-1] != ("ok", None)


def T():
    return ("fail", "transport")


OK = ("ok", None)


def gen(seed_hex):
    seed = bytes.fromhex(seed_hex)
    assert len(seed) == 32
    eps = []

    def add(stratum, kind, dims, seeds, script):
        if all(admitted(d) for d in dims):
            check_script(kind, script)
        lines = build_episode(kind, dims, seeds, script)
        eps.append((stratum, lines))

    def adm_dims(s):
        return (s.rng(1, 256), s.rng(1, 256), s.rng(1, 256))

    def seeds_for(s, kind):
        if kind == "mm":
            return (s.below(65536),)
        a = s.below(65536)
        b = s.below(65535)
        if b >= a:
            b += 1  # distinct seeds in every stratum except PAIRREPLAY
        return (a, b)

    # EDGE: every dim position x {0x000, 0x001, 0x100, 0x101, 0xfff} x kind.
    s = Stream(seed, "EDGE")
    for kind in ("mm", "pair"):
        for pos in range(3):
            for v in (0x000, 0x001, 0x100, 0x101, 0xfff):
                dims = list(adm_dims(s))
                dims[pos] = v
                sd = seeds_for(s, kind)
                script = [[OK]] * (1 if kind == "mm" else 2)
                add("EDGE", kind, tuple(dims), sd, script)

    # CODE: every non-transport C' code, mm, and pair at job 1 and job 2.
    s = Stream(seed, "CODE")
    for code in NONT:
        add("CODE", "mm", adm_dims(s), seeds_for(s, "mm"), [[res_for(code)]])
        add("CODE", "pair", adm_dims(s), seeds_for(s, "pair"), [[res_for(code)]])
        add("CODE", "pair", adm_dims(s), seeds_for(s, "pair"), [[OK], [res_for(code)]])

    # RETRY: transport retry patterns, three replicas each.
    s = Stream(seed, "RETRY")
    for _rep in range(3):
        mm_pats = [[[T(), OK]], [[T(), T()]]] + [[[T(), res_for(c)]] for c in NONT]
        for pat in mm_pats:
            add("RETRY", "mm", adm_dims(s), seeds_for(s, "mm"), pat)
        x1, x2 = s.pick(NONT), s.pick(NONT)
        pair_pats = [
            [[T(), OK], [OK]],
            [[OK], [T(), OK]],
            [[T(), OK], [T(), OK]],
            [[T(), T()]],
            [[OK], [T(), T()]],
            [[T(), OK], [T(), T()]],
            [[T(), res_for(x1)]],
            [[OK], [T(), res_for(x2)]],
        ]
        for pat in pair_pats:
            add("RETRY", "pair", adm_dims(s), seeds_for(s, "pair"), pat)

    # PAIRREPLAY: pair with equal seeds; job 2 is the same document -> refused replay.
    s = Stream(seed, "PAIRREPLAY")
    for _ in range(5):
        sv = s.below(65536)
        add("PAIRREPLAY", "pair", adm_dims(s), (sv, sv), [[OK], [res_for("replay")]])

    # XCLASS: grammar-legal R with the class the runtime never pairs with the code
    # (e.g. `res fail replay`, `res refused transport`). Policy: mirror; only
    # `res fail transport` triggers a retry. Reported separately, not thresholded.
    s = Stream(seed, "XCLASS")
    for i in range(10):
        code = RCODES[i % len(RCODES)]
        cls = "refused" if RUNTIME_CLASS[code] == "fail" else "fail"
        kind = "mm" if i % 2 == 0 else "pair"
        add("XCLASS", kind, adm_dims(s), seeds_for(s, kind), [[(cls, code)]])

    # RAND: 400 episodes from the declared distribution (EVAL-CONTRACT §3).
    s = Stream(seed, "RAND")
    for _ in range(400):
        kind = "mm" if s.below(2) == 0 else "pair"
        dims = list(adm_dims(s))
        if s.below(5) == 0:  # p = 0.2: out-of-bound goal
            nbad = 1 if s.below(4) else 2  # 3/4 one bad dim, 1/4 two
            positions = [0, 1, 2]
            for _b in range(nbad):
                pos = positions.pop(s.below(len(positions)))
                dims[pos] = 0 if s.below(8) == 0 else s.rng(257, 4095)
        sd = seeds_for(s, kind)
        script = []
        for _j in range(1 if kind == "mm" else 2):
            att = []
            for a in range(2):
                r = s.below(20)  # ok 12/20, transport 3/20, other 5/20
                if r < 12:
                    att.append(OK)
                elif r < 15:
                    att.append(T())
                else:
                    att.append(res_for(s.pick(NONT)))
                if att[-1] != T() or a == 1:
                    break
            script.append(att)
            if att[-1] != OK:
                break
        add("RAND", kind, tuple(dims), sd, script)
    return eps


def render(eps):
    out = []
    for i, (stratum, lines) in enumerate(eps):
        out.append("@ s%04d %s" % (i, stratum))
        out.append(EPISODE_HEADER)
        out.extend(lines)
    return "\n".join(out) + "\n"


def read_bundle(text):
    """Sealed bundle text -> list of (id, stratum, [message lines])."""
    eps = []
    for ln in text.rstrip("\n").split("\n"):
        if ln.startswith("@ "):
            _, eid, stratum = ln.split(" ")
            eps.append((eid, stratum, []))
        elif ln == EPISODE_HEADER:
            continue
        else:
            eps[-1][2].append(ln)
    return eps


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: oe_sealgen.py SEEDFILE OUTFILE\n")
        return 2
    seed_hex = open(argv[1]).read().strip()
    if len(seed_hex) != 64 or any(c not in "0123456789abcdef" for c in seed_hex):
        sys.stderr.write("seed must be 64 lowercase hex chars\n")
        return 2
    text = render(gen(seed_hex))
    with open(argv[2], "w", newline="\n") as f:
        f.write(text)
    print("%s episodes=%d sha256=%s" % (GENERATOR_ID, text.count("\n@ ") + 1,
                                        hashlib.sha256(text.encode()).hexdigest()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
