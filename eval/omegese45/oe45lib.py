"""Eval-owned R45 codec, grammar parser, binding check and reference policy.

Independent oracle for aien-train #45 (eval lane). Written from the shared
interface (docs: interface v1 sections 2-3), not from the contract, translator or
pipeline lanes' code. Python standard library only.

Two deliberately separate codings of the reference policy exist:
- `policy_next` (this file) decides the next model message from an episode
  prefix; it is the deterministic control.
- `oe_sealgen.build_episode` writes whole episodes from a per-job outcome
  script; it is the sealed-set generator.
The deterministic control scoring 1.0 on the sealed set cross-checks the two.
"""

VOCAB = (
    ["<pad>", "goal", "act", "res", "rep", "end", "mm", "pair", "job", "refuse",
     "ok", "refused", "fail", "bound", "replay", "budget", "worker", "transport",
     "receipt", "recomp", "verdict", "io", "ushort", "ulong", "0x"]
    + list("0123456789abcdef")
)
assert len(VOCAB) == 41
WID = {w: i for i, w in enumerate(VOCAB)}
HEX = "0123456789abcdef"
CODES = ["bound", "replay", "budget", "worker", "transport", "receipt", "recomp",
         "verdict", "io"]
RCODES = CODES[1:]  # C' = C without bound (runtime-only)
# Runtime class of each C' code as aienet0-run reports it (interface §2 and §5):
# exit 11 worker refusal -> refused (replay, budget, worker);
# exits 10, 12, 13, 14, 15 -> fail (transport, receipt, recomp, verdict, io).
RUNTIME_CLASS = {"replay": "refused", "budget": "refused", "worker": "refused",
                 "transport": "fail", "receipt": "fail", "recomp": "fail",
                 "verdict": "fail", "io": "fail"}
DIM_LO, DIM_HI = 0x001, 0x100
EPISODE_HEADER = "oe45 v1"


class CodecError(ValueError):
    pass


def surface_to_ids(line):
    """Strict surface text -> ids. Raises CodecError with a typed reason."""
    if isinstance(line, str):
        try:
            raw = line.encode("ascii")
        except UnicodeEncodeError:
            raise CodecError("non-ascii")
    else:
        raw = bytes(line)
    if any(b > 0x7E or (b < 0x20) for b in raw):
        if b"\r" in raw:
            raise CodecError("cr")
        if b"\t" in raw:
            raise CodecError("tab")
        raise CodecError("non-ascii-or-control")
    text = raw.decode("ascii")
    if text == "":
        raise CodecError("empty")
    words = text.split(" ")
    if any(w == "" for w in words):
        raise CodecError("space")
    ids = []
    for w in words:
        if w.startswith("0x") and len(w) > 2:
            ids.append(WID["0x"])
            for ch in w[2:]:
                if ch not in HEX:
                    raise CodecError("bad-hex")
                ids.append(WID[ch])
        elif w in WID and w != "<pad>" and w != "0x" and w not in HEX:
            ids.append(WID[w])
        else:
            raise CodecError("unknown-word")
    return ids


def ids_to_surface(ids):
    out = []
    i = 0
    while i < len(ids):
        t = ids[i]
        if not (0 <= t < len(VOCAB)) or t == 0:
            raise CodecError("bad-id")
        w = VOCAB[t]
        if w == "0x":
            j = i + 1
            s = "0x"
            while j < len(ids) and VOCAB[ids[j]] in HEX and len(VOCAB[ids[j]]) == 1:
                s += VOCAB[ids[j]]
                j += 1
            out.append(s)
            i = j
            continue
        out.append(w)
        i += 1
    return " ".join(out)


# ---------------------------------------------------------------- grammar
class Msg:
    """Parsed message. kind in G A R P; fields per grammar."""
    __slots__ = ("kind", "frame", "jk", "dims", "seeds", "cls", "code", "text")

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))

    def __repr__(self):
        return "Msg(%s)" % self.text


def _hexval(ids, pos, n):
    v = 0
    for t in ids[pos:pos + n]:
        w = VOCAB[t] if 0 <= t < len(VOCAB) else ""
        if len(w) != 1 or w not in HEX:
            return None
        v = v * 16 + HEX.index(w)
    if pos + n > len(ids):
        return None
    return v


def _dim(ids, p):
    # D := ushort 0x H H H  (5 ids)
    if p + 5 > len(ids) or ids[p] != WID["ushort"] or ids[p + 1] != WID["0x"]:
        return None
    return _hexval(ids, p + 2, 3)


def _seed(ids, p):
    # S := ulong 0x H H H H  (6 ids)
    if p + 6 > len(ids) or ids[p] != WID["ulong"] or ids[p + 1] != WID["0x"]:
        return None
    return _hexval(ids, p + 2, 4)


def _operands(ids, p, kind):
    dims = []
    for _ in range(3):
        d = _dim(ids, p)
        if d is None:
            return None, p
        dims.append(d)
        p += 5
    nseeds = 1 if kind == "mm" else 2
    seeds = []
    for _ in range(nseeds):
        s = _seed(ids, p)
        if s is None:
            return None, p
        seeds.append(s)
        p += 6
    return (tuple(dims), tuple(seeds)), p


def parse_ids(ids):
    """Parse one message (ids). Returns Msg or raises ValueError(reason)."""
    ids = list(ids)
    if not ids:
        raise ValueError("empty")
    W = WID
    f = ids[0]
    def done(p, m):
        if p >= len(ids) or ids[p] != W["end"]:
            raise ValueError("missing-end")
        if p + 1 != len(ids):
            raise ValueError("bytes-after-end")
        m.text = ids_to_surface(ids)
        return m
    if f == W["goal"]:
        if len(ids) < 2 or ids[1] not in (W["mm"], W["pair"]):
            raise ValueError("goal-kind")
        kind = VOCAB[ids[1]]
        ops, p = _operands(ids, 2, kind)
        if ops is None:
            raise ValueError("goal-operand")
        return done(p, Msg(kind="G", frame="goal", jk=kind, dims=ops[0], seeds=ops[1]))
    if f == W["act"]:
        if len(ids) >= 2 and ids[1] == W["job"]:
            if len(ids) < 3 or ids[2] != W["mm"]:
                raise ValueError("act-kind")
            ops, p = _operands(ids, 3, "mm")
            if ops is None:
                raise ValueError("act-operand")
            return done(p, Msg(kind="A", frame="act", jk="job", dims=ops[0], seeds=ops[1]))
        if len(ids) >= 3 and ids[1] == W["refuse"] and ids[2] == W["bound"]:
            return done(3, Msg(kind="A", frame="act", jk="refuse", code="bound"))
        raise ValueError("act-form")
    if f == W["res"]:
        if len(ids) >= 2 and ids[1] == W["ok"]:
            return done(2, Msg(kind="R", frame="res", cls="ok"))
        if len(ids) >= 3 and ids[1] in (W["refused"], W["fail"]) and VOCAB[ids[2]] in RCODES:
            return done(3, Msg(kind="R", frame="res", cls=VOCAB[ids[1]], code=VOCAB[ids[2]]))
        raise ValueError("res-form")
    if f == W["rep"]:
        if len(ids) >= 2 and ids[1] == W["ok"]:
            if len(ids) < 3 or ids[2] not in (W["mm"], W["pair"]):
                raise ValueError("rep-kind")
            kind = VOCAB[ids[2]]
            ops, p = _operands(ids, 3, kind)
            if ops is None:
                raise ValueError("rep-operand")
            return done(p, Msg(kind="P", frame="rep", cls="ok", jk=kind, dims=ops[0], seeds=ops[1]))
        if len(ids) >= 3 and ids[1] in (W["refused"], W["fail"]) and ids[2] < len(VOCAB) and VOCAB[ids[2]] in CODES:
            return done(3, Msg(kind="P", frame="rep", cls=VOCAB[ids[1]], code=VOCAB[ids[2]]))
        raise ValueError("rep-form")
    raise ValueError("frame")


def parse_line(line):
    return parse_ids(surface_to_ids(line))


def is_model_msg(m):
    return m is not None and m.kind in ("A", "P")


def admitted(d):
    return DIM_LO <= d <= DIM_HI


# ------------------------------------------------------------- formatting
def fmt_dim(d):
    return "ushort 0x%03x" % d


def fmt_seed(s):
    return "ulong 0x%04x" % s


def fmt_ops(dims, seeds):
    return " ".join([fmt_dim(d) for d in dims] + [fmt_seed(s) for s in seeds])


def fmt_goal(kind, dims, seeds):
    return "goal %s %s end" % (kind, fmt_ops(dims, seeds))


def fmt_act(dims, seed):
    return "act job mm %s end" % fmt_ops(dims, (seed,))


ACT_REFUSE = "act refuse bound end"


def fmt_res(cls, code=None):
    return "res ok end" if cls == "ok" else "res %s %s end" % (cls, code)


def fmt_rep_ok(kind, dims, seeds):
    return "rep ok %s %s end" % (kind, fmt_ops(dims, seeds))


def fmt_rep(cls, code):
    return "rep %s %s end" % (cls, code)


# ------------------------------------------------------- binding check
def binding_ok(goal, act, job_index):
    """Interface §3 binding: act job dims == goal dims, seed == current job seed,
    every dim admitted. Returns (ok, reason)."""
    if act.kind != "A" or act.jk != "job":
        return True, "not-a-job"
    if any(not admitted(d) for d in act.dims):
        return False, "bound"
    if act.dims != goal.dims:
        return False, "mismatch-dims"
    if job_index >= len(goal.seeds) or act.seeds[0] != goal.seeds[job_index]:
        return False, "mismatch-seed"
    return True, "ok"


def current_job_index(msgs):
    """Index of the job the next action addresses, given the prefix (interface §3.2)."""
    goal = msgs[0]
    j, tseen = 0, False
    for m in msgs[1:]:
        if m.kind != "R":
            continue
        if m.cls == "ok":
            j, tseen = j + 1, False
        elif m.cls == "fail" and m.code == "transport" and not tseen:
            tseen = True
        else:
            return None
    return j if j < len(goal.seeds) else None


# ----------------------------------------------------- reference policy
def policy_next(msgs):
    """Deterministic reference policy (interface §3): next model message text
    from a parsed episode prefix [G, (A, R?)*]. Uses only G and the R messages."""
    goal = msgs[0]
    assert goal.kind == "G"
    if any(not admitted(d) for d in goal.dims):
        if len(msgs) == 1:
            return ACT_REFUSE
        return fmt_rep("refused", "bound")
    njobs = len(goal.seeds)
    j, tseen = 0, False
    for m in msgs[1:]:
        if m.kind != "R":
            continue
        if m.cls == "ok":
            j, tseen = j + 1, False
            continue
        if m.cls == "fail" and m.code == "transport" and not tseen:
            tseen = True
            continue
        return fmt_rep(m.cls, m.code)
    if j >= njobs:
        return fmt_rep_ok(goal.jk, goal.dims, goal.seeds)
    return fmt_act(goal.dims, goal.seeds[j])


# ----------------------------------------------------------- episodes
def parse_episode(text):
    """OE45 v1 episode text -> list of lines (messages). Raises ValueError."""
    if not text.endswith("\n"):
        raise ValueError("no-final-lf")
    lines = text[:-1].split("\n")
    if lines[0] != EPISODE_HEADER:
        raise ValueError("header")
    return lines[1:]


def check_episode_shape(msgs):
    """Sequence G (A R?)+ P; an act refuse gets no R. Returns reason or None."""
    if not msgs or msgs[0].kind != "G":
        return "first-not-G"
    i = 1
    nact = 0
    while i < len(msgs) and msgs[i].kind == "A":
        a = msgs[i]
        nact += 1
        i += 1
        if a.jk == "refuse":
            continue
        if i >= len(msgs) or msgs[i].kind != "R":
            return "act-without-res"
        i += 1
    if nact == 0:
        return "no-act"
    if i != len(msgs) - 1 or msgs[i].kind != "P":
        return "not-ending-P"
    return None


def decode_points(msgs):
    """Indices i such that msgs[i] is a model message (A or P)."""
    return [i for i, m in enumerate(msgs) if m.kind in ("A", "P")]


def episode_ids_len(lines):
    return sum(len(surface_to_ids(l)) for l in lines)
