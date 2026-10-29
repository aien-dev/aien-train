#!/usr/bin/env python3
"""Controls and metrics on the sealed set (EVAL-CONTRACT.md sections 4-5).

Eval-owned (OE45Eval). Python standard library only. Ground truth is the
sealed bundle (oe_sealgen.py) and the eval reference policy (oe45lib.py);
the pipeline's generator output (train/dev) is read only for the retrieval
control, the overlap and leakage checks and the spec audit.

Model controls run the pipeline core through its documented CLI
(docs/omegese45/CORE-CLI.md: `batch CKDIR U|G IN OUT`), one batch per mode
and checkpoint, over every decode-point prefix of the sealed set.

Closed-loop success is computed from the teacher-forced decodes: every
control here is a deterministic function of its input prefix (argmax decode,
reference policy) or of the decode-point index (retrieval), and the scripted
R messages are fixed, so while every earlier model message equals the
reference the closed-loop prefix IS the sealed prefix. An episode therefore
succeeds in closed loop iff every one of its decode points is an exact
match, and the first mismatch ends it, as the contract defines.

Usage: oe_controls.py SEALED SEED TRAIN DEV CORE CK0 CK1 OUTDIR
Use '-' for CORE/CK0/CK1 to run DET/RET only (model controls print NOT_RUN).
Writes counts.json, intervals.json, decodes.tsv, overlap.txt, xclass.tsv,
spec-audit.txt, hash-check.txt and summary.txt in OUTDIR.
"""

import json
import hashlib
from collections import Counter
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import oe45lib as L  # noqa: E402
from oe_audit import audit_one  # noqa: E402
from oe_sealgen import read_bundle  # noqa: E402
import oe_episode as EP
from oe_wilson import wilson

HEADER = "oe45 v1"
REFUSE = "act refuse bound end"
TRANSPORT = "res fail transport end"


def split_corpus(text):
    """Pipeline generator output: concatenated `oe45 v1` episodes -> [lines]."""
    eps = []
    for ln in text.rstrip("\n").split("\n"):
        if ln == HEADER:
            eps.append([])
        else:
            eps[-1].append(ln)
    return eps


def model_lines(lines):
    msgs = [L.parse_line(x) for x in lines]
    return [lines[i] for i in L.decode_points(msgs)]


def nearest(goal_ids, train_goals):
    """Index of the train episode with the smallest position-wise id Hamming
    distance (missing positions count as mismatches; first in order on ties)."""
    best, best_d = -1, None
    gl = len(goal_ids)
    for idx, ids in train_goals:
        d = abs(len(ids) - gl) + sum(a != b for a, b in zip(ids, goal_ids))
        if best_d is None or d < best_d:
            best, best_d = idx, d
            if d == 0:
                break
    return best, best_d


def run_batch(core, ck, mode, prefixes, path):
    inp = path + ".in"
    with open(inp, "w") as f:
        f.write("".join(prefixes))
    r = subprocess.run(["taskset", "-c", "10-14", "nice", "-n", "19", "ionice", "-c3",
                        core, "batch", ck, mode, inp, path], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("core batch %s %s rc=%d %s" % (ck, mode, r.returncode, r.stderr))
    out = open(path).read().rstrip("\n").split("\n")
    if len(out) != len(prefixes):
        raise SystemExit("core batch %s %s: %d lines for %d prefixes" % (ck, mode, len(out), len(prefixes)))
    res = []
    for ln in out:
        res.append(ln[3:] if ln.startswith("ok ") else None)
    return res, out


def valid_model(text):
    if text is None:
        return False
    try:
        return L.parse_line(text).kind in ("A", "P")
    except ValueError:
        return False


def main(argv):
    if len(argv) != 9:
        sys.stderr.write(__doc__)
        return 2
    sealed_p, seed_p, train_p, dev_p, core, ck0, ck1, out = argv[1:]
    os.makedirs(out, exist_ok=True)
    pins = [
        (sealed_p, "1dc394c1cb9903572bf9b5f68532f18a80985ca2843faf67129607bb65dbce70"),
        (seed_p, "1adc002cf1c1df90c76d473783f4828112f1ec27210ad662d8f4cc2b9c3c82f2"),
        (os.path.join(HERE, "oe_sealgen.py"), "38fbeb7ced137d1184eb9aa9c1f42e6dfcec90189f0e6729623af4419a7d812a"),
        (os.path.join(HERE, "oe45lib.py"), "31ea114aae4b8d8c68f8c3834c4b5e9f65db9d7ac0195d042b35de2f186a326d"),
    ]
    hashes = []
    for path, expected in pins:
        observed = hashlib.sha256(open(path, "rb").read()).hexdigest()
        hashes.append("%s %s %s" % ("PASS" if observed == expected else "FAIL", observed, path))
    open(os.path.join(out, "hash-check.txt"), "w").write("\n".join(hashes) + "\n")
    if any(x.startswith("FAIL") for x in hashes):
        print("INVALID commitment hash mismatch")
        return 1
    sealed = read_bundle(open(sealed_p).read())
    train = split_corpus(open(train_p).read())
    dev = split_corpus(open(dev_p).read())
    manifest = {}
    artifacts = [sealed_p, seed_p, train_p, dev_p]
    if core != "-":
        artifacts.append(core)
    for ck in (ck0, ck1):
        if ck != "-":
            artifacts.extend(os.path.join(ck, name) for name in ("ck.a", "ck.b", "ck.done")
                             if os.path.isfile(os.path.join(ck, name)))
    for path in artifacts:
        manifest[path] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    json.dump(manifest, open(os.path.join(out, "inputs-sha256.json"), "w"), indent=1, sort_keys=True)
    for name, ck in (("ck0", ck0), ("ck1", ck1)):
        if core != "-" and ck != "-":
            r = subprocess.run([core, "info", ck], capture_output=True, text=True)
            open(os.path.join(out, name + "-info.txt"), "w").write(
                r.stdout + r.stderr + "RC %d\n" % r.returncode)
            if r.returncode:
                raise ValueError("%s checkpoint info failed" % name)

    # ---- spec audit of the pipeline generator (contract 4.6)
    spec = []
    spec_bad = False
    for name, corpus in (("train", train), ("dev", dev),
                         ("sealed", [lines for _, _, lines in sealed])):
        bad = []
        for i, lines in enumerate(corpus):
            ok, why, _ = audit_one(lines)
            if not ok:
                bad.append((i, why))
        spec_bad = spec_bad or bool(bad)
        spec.append("%s episodes=%d disagreements=%d" % (name, len(corpus), len(bad)))
        spec.extend("  %s[%d] %s" % (name, i, w) for i, w in bad[:50])
    open(os.path.join(out, "spec-audit.txt"), "w").write("\n".join(spec) + "\n")

    # ---- overlap and leakage (contract 4.2-4.4, 4.7)
    tg = {e[0] for e in train}
    dg = {e[0] for e in dev}
    te = {"\n".join(e) for e in train}
    de = {"\n".join(e) for e in dev}
    train_r = Counter(l for e in train for l in e if l.startswith("res "))
    xc_report = ["eid\tR\ttrain_R_count\tfull_sequence_in_train\tfull_sequence_in_dev"]
    ov = {"goal_in_train": 0, "goal_in_dev": 0, "episode_in_train": 0, "episode_in_dev": 0,
          "episode_overlap_without_goal_overlap": 0, "xclass_lines_in_train": 0}
    disjoint = set()
    xlines = []
    for eid, stratum, lines in sealed:
        g_t, g_d = lines[0] in tg, lines[0] in dg
        e_t, e_d = "\n".join(lines) in te, "\n".join(lines) in de
        ov["goal_in_train"] += g_t
        ov["goal_in_dev"] += g_d
        ov["episode_in_train"] += e_t
        ov["episode_in_dev"] += e_d
        ov["episode_overlap_without_goal_overlap"] += (e_t and not g_t) or (e_d and not g_d)
        if not g_t and not g_d:
            disjoint.add(eid)
        if stratum == "XCLASS":
            for l in lines:
                m = L.parse_line(l)
                if m.kind == "R" and m.code and L.RUNTIME_CLASS.get(m.code) != m.cls:
                    xlines.append(l)
                    ov["xclass_lines_in_train"] += l in train_r
                    xc_report.append("%s\t%s\t%d\t%s\t%s" % (eid, l, train_r[l], e_t, e_d))
    ov["xclass_lines"] = len(xlines)
    ov["goal_disjoint_thresholded"] = sum(1 for e, s, _ in sealed if e in disjoint and s != "XCLASS")
    # Parent clarification 2026-10-29: public grammar R strings are not
    # sealed-only data. XCLASS line exposure is reported, never called novel.
    leak = ov["episode_overlap_without_goal_overlap"] > 0
    ov["leakage"] = leak
    open(os.path.join(out, "overlap.txt"), "w").write(
        "".join("%s %s\n" % (k, v) for k, v in ov.items()))
    open(os.path.join(out, "xclass.tsv"), "w").write("\n".join(xc_report) + "\n")

    # ---- decode points
    points = []  # (eid, stratum, k, i, prefix_text, reference, msgs)
    for eid, stratum, lines in sealed:
        msgs = [L.parse_line(x) for x in lines]
        for k, i in enumerate(L.decode_points(msgs)):
            points.append((eid, stratum, k, i, HEADER + "\n" + "".join(x + "\n" for x in lines[:i]),
                           lines[i], msgs, lines))
    prefixes = [p[4] for p in points]

    # ---- controls
    emitted = {}
    raw = {}
    emitted["DET"] = [L.policy_next(p[6][:p[3]]) for p in points]
    train_goals = []
    seen = set()
    for idx, e in enumerate(train):
        if e[0] not in seen:  # first occurrence wins ties anyway
            seen.add(e[0])
            train_goals.append((idx, L.surface_to_ids(e[0])))
    ret_pick = {}
    for eid, stratum, lines in sealed:
        ret_pick[eid] = nearest(L.surface_to_ids(lines[0]), train_goals)
    emitted["RET"] = []
    for p in points:
        ml = model_lines(train[ret_pick[p[0]][0]])
        emitted["RET"].append(ml[p[2]] if p[2] < len(ml) else None)
    for name, ck, mode in (("U0", ck0, "U"), ("G0", ck0, "G"), ("U1", ck1, "U"), ("G1", ck1, "G")):
        if core == "-" or ck == "-":
            print("%s NOT_RUN checkpoint/core not supplied" % name)
            continue
        emitted[name], raw[name] = run_batch(core, ck, mode, prefixes, os.path.join(out, "batch-%s.out" % name))

    with open(os.path.join(out, "decodes.tsv"), "w") as f:
        f.write("eid\tstratum\tk\treference\t" + "\t".join(emitted) + "\n")
        for j, p in enumerate(points):
            f.write("\t".join([p[0], p[1], str(p[2]), p[5]] + [str(emitted[c][j]) for c in emitted]) + "\n")

    # ---- metrics
    def counts(ctrl, keep):
        em = emitted[ctrl]
        c = {k: [0, 0] for k in ("validity", "action_exact", "closed_loop", "completion", "recovery",
                                 "refusal_precision", "refusal_recall", "retry", "report_exact")}
        c["binding_needs"] = 0
        c["binding_opportunities"] = 0
        ep_ok = {}
        for j, p in enumerate(points):
            eid, stratum, k, i, _, ref, msgs, lines = p
            if not keep(eid, stratum):
                continue
            got = em[j]
            match = got == ref
            c["validity"][0] += valid_model(got)
            c["validity"][1] += 1
            c["action_exact"][0] += match
            c["action_exact"][1] += 1
            ep_ok[eid] = ep_ok.get(eid, True) and match
            if lines[i - 1] == TRANSPORT:
                c["retry"][0] += match
                c["retry"][1] += 1
            if msgs[i].kind == "P":
                c["report_exact"][0] += match
                c["report_exact"][1] += 1
            if k == 0:
                pos, ref_pos = got == REFUSE, ref == REFUSE
                c["refusal_precision"][0] += pos and ref_pos
                c["refusal_precision"][1] += pos
                c["refusal_recall"][0] += pos and ref_pos
                c["refusal_recall"][1] += ref_pos
            if valid_model(got):
                m = L.parse_line(got)
                if m.kind == "A" and m.jk == "job":
                    c["binding_opportunities"] += 1
                    ok, _ = EP.bind_current(msgs[0], m, msgs[:i])
                    c["binding_needs"] += not ok
        for eid, stratum, lines in sealed:
            if eid not in ep_ok:
                continue
            msgs = [L.parse_line(x) for x in lines]
            s = ep_ok[eid]
            c["closed_loop"][0] += s
            c["closed_loop"][1] += 1
            if msgs[-1].kind == "P" and msgs[-1].cls == "ok":
                c["completion"][0] += s
                c["completion"][1] += 1
            admitted = lines[1] != REFUSE
            if admitted and any(m.kind == "R" and m.cls != "ok" for m in msgs):
                c["recovery"][0] += s
                c["recovery"][1] += 1
        return c

    subsets = {
        "thresholded": lambda e, s: s != "XCLASS",
        "goal_disjoint": lambda e, s: s != "XCLASS" and e in disjoint,
        "xclass": lambda e, s: s == "XCLASS",
    }
    result = {"overlap": ov, "controls": {}}
    for ctrl in emitted:
        result["controls"][ctrl] = {sub: counts(ctrl, f) for sub, f in subsets.items()}
    result["ret_mean_distance"] = [sum(d for _, d in ret_pick.values()), len(ret_pick)]
    result["ret_exact_goal"] = sum(1 for _, d in ret_pick.values() if d == 0)
    json.dump(result, open(os.path.join(out, "counts.json"), "w"), indent=1, sort_keys=True)
    intervals = {}
    detail = ["subset\tcontrol\tmetric\tk\tn\tWilson95_lower\tWilson95_upper"]
    for ctrl, subs in result["controls"].items():
        intervals[ctrl] = {}
        for sub, c in subs.items():
            intervals[ctrl][sub] = {}
            metrics = {k: v for k, v in c.items() if isinstance(v, list)}
            metrics["binding_needed_rate"] = [c["binding_needs"], c["binding_opportunities"]]
            for metric, (k, n) in metrics.items():
                ci = wilson(k, n)
                intervals[ctrl][sub][metric] = ci
                lo, hi = ("NOT_DEFINED", "NOT_DEFINED") if ci is None else (
                    "%.12g" % ci[0], "%.12g" % ci[1])
                detail.append("%s\t%s\t%s\t%d\t%d\t%s\t%s" % (sub, ctrl, metric, k, n, lo, hi))
    json.dump(intervals, open(os.path.join(out, "intervals.json"), "w"), indent=1, sort_keys=True)
    open(os.path.join(out, "intervals.tsv"), "w").write("\n".join(detail) + "\n")

    lines = ["subset control " + " ".join(result["controls"]["DET"]["thresholded"])]
    for sub in subsets:
        for ctrl in emitted:
            c = result["controls"][ctrl][sub]
            lines.append("%s %s %s" % (sub, ctrl, " ".join(
                "%d/%d" % tuple(v) if isinstance(v, list) else str(v) for v in c.values())))
    lines.extend("%s NOT_RUN checkpoint/core not supplied" % ctrl
                 for ctrl in ("U0", "G0", "U1", "G1") if ctrl not in emitted)
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n".join(detail))
    print("".join("%s %s\n" % (k, v) for k, v in ov.items()), end="")
    print(spec[0])
    print(spec[[i for i, s in enumerate(spec) if s.startswith("dev ")][0]])
    det_bad = any(k != n for k, n in (
        v for v in result["controls"]["DET"]["thresholded"].values() if isinstance(v, list)))
    invalid = det_bad or leak or spec_bad
    print("MACHINERY %s" % ("INVALID" if invalid else "PASS"))
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
