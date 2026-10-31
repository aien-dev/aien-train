"""Eval-owned parser of the rep2en English response (aien-train #45).

The sentence shapes are transcribed from the translator's documented
fixed templates: docs/omegese45/TRANSLATOR.md section "rep2en" (tables
"Templates" and "Plain words") at feat/45-chat-translator cdd1a4b9. This
module does not
import or run translator code. It parses one response into the facts it
states, then checks every stated fact against the runtime facts.tsv rows and
the episode, which check-demo has already bound to the signed artifacts.

Shapes (numbers are decimal without leading zeros; hashes are 64 lowercase
hex digits; one line, final LF):
  OK mm:   Completed and verified: one matmul_u32 job multiplying a M by K
           matrix by a K by N matrix with seed S; job 1 dag_id D mjr1_sha256 H.
  OK pair: Completed and verified: two matmul_u32 jobs, each multiplying a M
           by K matrix by a K by N matrix, with seeds S and T; job 1 dag_id D1
           mjr1_sha256 H1; job 2 dag_id D2 mjr1_sha256 H2.
  bound:   Not completed: refused before dispatch (code bound): the goal asks
           for a M by K by N matmul and a dimension is outside the admitted
           range 1 to 256; nothing was dispatched.
  stop:    Not completed: job J was refused (code C): REASON.
           Not completed: job J failed (code C): REASON.
"""

import re

NUM = r"(0|[1-9][0-9]*)"
HEX = r"([0-9a-f]{64})"
JOBREF = r"job %s dag_id %s mjr1_sha256 %s" % (NUM, HEX, HEX)
DIMS = r"a %s by %s matrix by a %s by %s matrix" % (NUM, NUM, NUM, NUM)

RE_MM = re.compile(r"Completed and verified: one matmul_u32 job multiplying %s with seed %s; %s\.\n\Z"
                   % (DIMS, NUM, JOBREF))
RE_PAIR = re.compile(r"Completed and verified: two matmul_u32 jobs, each multiplying %s, with seeds %s and %s; %s; %s\.\n\Z"
                     % (DIMS, NUM, NUM, JOBREF, JOBREF))
RE_BOUND = re.compile(r"Not completed: refused before dispatch \(code bound\): the goal asks for a %s by %s by %s matmul"
                      r" and a dimension is outside the admitted range 1 to 256; nothing was dispatched\.\n\Z"
                      % (NUM, NUM, NUM))
RE_STOP = re.compile(r"Not completed: job %s (was refused|failed) \(code ([a-z]+)\): ([a-z ]+)\.\n\Z" % NUM)

# Reason phrase that each code's sentence must carry (one phrase per code; a
# phrase naming a different failure than the code is UNFAITHFUL).
REASON = {
    "replay": "the worker refused a replay of an already executed job",
    "budget": "the job was refused for budget or fuel",
    "worker": "the worker refused the job",
    "transport": "the connection to the worker failed",
    "receipt": "the verifier refused the worker receipt",
    "recomp": "the verifier recompute failed",
    "verdict": "the recomputed result did not match the worker result",
    "io": "a local input or output error occurred",
}
CLASS_OF_VERB = {"was refused": "refused", "failed": "fail"}


def parse(text):
    """English -> claim dict, or None when no documented shape matches."""
    m = RE_MM.match(text)
    if m:
        g = m.groups()
        return {"shape": "ok", "kind": "mm", "dims": (int(g[0]), int(g[1]), int(g[3])),
                "k2": int(g[2]), "seeds": (int(g[4]),), "jobs": [(int(g[5]), g[6], g[7])]}
    m = RE_PAIR.match(text)
    if m:
        g = m.groups()
        return {"shape": "ok", "kind": "pair", "dims": (int(g[0]), int(g[1]), int(g[3])),
                "k2": int(g[2]), "seeds": (int(g[4]), int(g[5])),
                "jobs": [(int(g[6]), g[7], g[8]), (int(g[9]), g[10], g[11])]}
    m = RE_BOUND.match(text)
    if m:
        return {"shape": "bound", "dims": tuple(int(x) for x in m.groups())}
    m = RE_STOP.match(text)
    if m:
        return {"shape": "stop", "job": int(m.group(1)), "cls": CLASS_OF_VERB[m.group(2)],
                "code": m.group(3), "reason": m.group(4)}
    return None


def last_row(rows, job):
    got = [r for r in rows if r["job"] == job]
    return got[-1] if got else None


def check_response(text, msgs, rows):
    """Yield (name, status, detail) checks of the English against facts and episode."""
    out = []

    def rec(name, ok, detail=""):
        out.append((name, "PASS" if ok else "FAIL", detail))

    claim = parse(text)
    if claim is None:
        rec("english.parse", False, "no documented shape: %r" % text[:120])
        return out
    rec("english.parse", True, claim["shape"])
    goal, rep = msgs[0], msgs[-1]
    gdims = tuple(goal.dims)
    if claim["shape"] == "ok":
        rec("english.kind", claim["kind"] == goal.jk and rep.cls == "ok" and rep.jk == goal.jk
            and rep.dims == goal.dims and rep.seeds == goal.seeds,
            "english=%s goal=%s report=%s" % (claim["kind"], goal.jk, rep.text))
        rec("english.dims", claim["dims"] == gdims and claim["k2"] == gdims[1],
            "english=%s k2=%d goal=%s" % (claim["dims"], claim["k2"], gdims))
        rec("english.seeds", claim["seeds"] == tuple(goal.seeds),
            "english=%s goal=%s" % (claim["seeds"], tuple(goal.seeds)))
        want_jobs = len(goal.seeds)
        rec("english.job_count", [j for j, _, _ in claim["jobs"]] == list(range(1, want_jobs + 1)))
        for j, dag, sha in claim["jobs"]:
            r = last_row(rows, j)
            ok = (r is not None and r["exit"] == 0 and r["class"] == "ok"
                  and r["dag_id"] == dag and r["mjr1_sha256"] == sha
                  and (r["m"], r["k"], r["n"]) == claim["dims"] and r["seed"] == claim["seeds"][j - 1])
            rec("english.job%d_receipt" % j, ok, "dag_id=%s mjr1_sha256=%s" % (dag, sha))
    elif claim["shape"] == "bound":
        rec("english.dims", claim["dims"] == gdims, "english=%s goal=%s" % (claim["dims"], gdims))
        rec("english.bound_true", any(not (1 <= d <= 256) for d in gdims), "goal=%s" % (gdims,))
        rec("english.nothing_dispatched", not rows, "rows=%d" % len(rows))
        rec("english.report", rep.text == "rep refused bound end", rep.text)
    else:
        r = rows[-1] if rows else None
        rec("english.stop_row", r is not None and r["job"] == claim["job"]
            and r["class"] == claim["cls"] and r["code"] == claim["code"],
            "english job=%d %s %s" % (claim["job"], claim["cls"], claim["code"]))
        rec("english.reason", REASON.get(claim["code"]) == claim["reason"], claim["reason"])
        rec("english.report", rep.cls == claim["cls"] and rep.code == claim["code"], rep.text)
    return out
