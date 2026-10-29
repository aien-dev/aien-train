#!/usr/bin/env python3
"""Audit an OE45 episode collection against interface v1.1 (eval-owned oracle).

Usage:
  oe_audit.py bundle FILE        sealed bundle (`@ id stratum` + episodes)
  oe_audit.py dir DIR            directory of OE45 v1 episode files (*.oe45 or all files)
  oe_audit.py files F1 F2 ...    individual episode files

For every episode: strict codec, grammar, sequence shape G (A R?)+ P, length
<= MAXLEN 176, and agreement of every model message with the eval reference
policy (policy_next) given the scripted R messages. Prints a coverage summary
and one line per disagreement. Exit 0 iff every episode passes.
Used both to self-check the sealed set and to check the pipeline generator's
train/dev output against the spec.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oe45lib import (CODES, EPISODE_HEADER, RCODES, admitted, check_episode_shape,
                     parse_episode, parse_line, policy_next, surface_to_ids)
from oe_sealgen import read_bundle

MAXLEN = 176


def load(mode, args):
    eps = []
    if mode == "bundle":
        for eid, stratum, lines in read_bundle(open(args[0]).read()):
            eps.append((eid, stratum, lines))
    else:
        files = []
        if mode == "dir":
            for root, _d, fs in os.walk(args[0]):
                for f in sorted(fs):
                    files.append(os.path.join(root, f))
            files.sort()
        else:
            files = args
        for p in files:
            try:
                lines = parse_episode(open(p, newline="").read())
            except (ValueError, UnicodeDecodeError) as e:
                eps.append((p, "?", None, str(e)))
                continue
            eps.append((p, "-", lines))
    return eps


def audit_one(lines):
    """Returns (ok, reason, info)."""
    try:
        msgs = [parse_line(l) for l in lines]
    except ValueError as e:
        return False, "grammar:%s" % e, None
    shape = check_episode_shape(msgs)
    if shape:
        return False, "shape:" + shape, None
    n = sum(len(surface_to_ids(l)) for l in lines)
    if n > MAXLEN:
        return False, "maxlen:%d" % n, None
    for i, m in enumerate(msgs):
        if m.kind in ("A", "P"):
            want = policy_next(msgs[:i])
            if m.text != want:
                return False, "policy@%d got[%s] want[%s]" % (i, m.text, want), None
    return True, "ok", (msgs, n)


def main(argv):
    if len(argv) < 3 or argv[1] not in ("bundle", "dir", "files"):
        sys.stderr.write(__doc__)
        return 2
    eps = load(argv[1], argv[2:])
    bad = 0
    cov = {"episodes": 0, "mm": 0, "pair": 0, "oob": 0, "retry": 0, "transport2": 0,
           "rep_ok": 0, "maxlen_ids": 0}
    rcov = {}
    edge = set()
    strata = {}
    for e in eps:
        eid, stratum, lines = e[0], e[1], e[2]
        if lines is None:
            bad += 1
            print("BAD %s format:%s" % (eid, e[3]))
            continue
        ok, why, info = audit_one(lines)
        cov["episodes"] += 1
        strata[stratum] = strata.get(stratum, 0) + 1
        if not ok:
            bad += 1
            print("BAD %s %s" % (eid, why))
            continue
        msgs, n = info
        g = msgs[0]
        cov[g.jk] += 1
        cov["maxlen_ids"] = max(cov["maxlen_ids"], n)
        for d in g.dims:
            if d in (0x000, 0x001, 0x100, 0x101):
                edge.add(d)
        if any(not admitted(d) for d in g.dims):
            cov["oob"] += 1
        nt = 0
        for m in msgs:
            if m.kind == "R" and m.cls != "ok":
                rcov["%s %s" % (m.cls, m.code)] = rcov.get("%s %s" % (m.cls, m.code), 0) + 1
                if m.code == "transport" and m.cls == "fail":
                    nt += 1
        if nt:
            cov["retry"] += 1
        if any(m.kind == "R" and m.cls == "fail" and m.code == "transport" for m in msgs) and \
                msgs[-1].text == "rep fail transport end":
            cov["transport2"] += 1
        if msgs[-1].cls == "ok":
            cov["rep_ok"] += 1
    print("COVERAGE " + " ".join("%s=%d" % kv for kv in cov.items()))
    print("STRATA " + " ".join("%s=%d" % kv for kv in sorted(strata.items())))
    print("RCODES " + " ".join("%s=%d" % kv for kv in sorted(rcov.items())))
    print("EDGES " + " ".join("0x%03x" % d for d in sorted(edge)))
    missing = [c for c in RCODES if not any(k.endswith(" " + c) for k in rcov)]
    print("MISSING_RCODES %s" % (" ".join(missing) if missing else "-"))
    print("AUDIT %s episodes=%d bad=%d" % ("PASS" if bad == 0 else "FAIL", len(eps), bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
