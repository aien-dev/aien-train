#!/usr/bin/env python3
"""Eval-lane self-tests (aien-train #45). Python standard library only.

1. Spec cases: episodes typed by hand from interface v1.1 section 3 must equal
   the eval generator's output and pass the audit; hand-written violations of
   the reference policy, grammar and codec must be rejected.
2. Sealed-generator determinism (same seed -> same bytes).
3. P-256 and layout readers against the REAL Rust-signed baseline artifacts
   (aien-protocols e705fdc9 evidence/aienet0-20261028/exercised/artifacts),
   including an independent matmul_u32 recompute of the stored digest.
4. check-demo end to end on synthetic case dirs, and its negative controls:
   tampered receipt byte, tampered output digest, edited English number,
   wrong English reason, unbound dispatch.
Prints one `T <name> PASS|FAIL` line per test and `OE45_EVAL_TESTS PASS|FAIL`.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EVAL = os.environ.get("OE45_EVAL_DIR") or os.path.join(ROOT, "eval", "omegese45")  # mutants override
sys.path.insert(0, EVAL)
sys.path.insert(0, HERE)

import oe45lib as L  # noqa: E402
import oe_an0 as A  # noqa: E402
import oe_audit  # noqa: E402
import oe_en  # noqa: E402
import oe_sealgen  # noqa: E402
import mkcase  # noqa: E402

FAILS = []


def t(name, cond, detail=""):
    print("T %s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAILS.append(name)


T_ = ("fail", "transport")
OK = ("ok", None)

# --------------------------------------------------------------- 1. spec cases
# Typed by hand from interface v1.1 section 3; not produced by any fmt_ helper.
SPEC = [
    ("mm_ok", "mm", (0x40, 0x40, 0x40), (0x1234,), [[OK]], [
        "goal mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1234 end",
        "act job mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1234 end",
        "res ok end",
        "rep ok mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1234 end"]),
    ("mm_edge_0x100_admitted", "mm", (0x100, 0x001, 0x100), (0xffff,), [[OK]], [
        "goal mm ushort 0x100 ushort 0x001 ushort 0x100 ulong 0xffff end",
        "act job mm ushort 0x100 ushort 0x001 ushort 0x100 ulong 0xffff end",
        "res ok end",
        "rep ok mm ushort 0x100 ushort 0x001 ushort 0x100 ulong 0xffff end"]),
    ("mm_bound_0x101", "mm", (0x101, 0x004, 0x004), (0x0007,), [], [
        "goal mm ushort 0x101 ushort 0x004 ushort 0x004 ulong 0x0007 end",
        "act refuse bound end",
        "rep refused bound end"]),
    ("pair_bound_0x000", "pair", (0x004, 0x000, 0x004), (0x0001, 0x0002), [], [
        "goal pair ushort 0x004 ushort 0x000 ushort 0x004 ulong 0x0001 ulong 0x0002 end",
        "act refuse bound end",
        "rep refused bound end"]),
    ("mm_replay", "mm", (0x002, 0x003, 0x004), (0x00aa,), [[("refused", "replay")]], [
        "goal mm ushort 0x002 ushort 0x003 ushort 0x004 ulong 0x00aa end",
        "act job mm ushort 0x002 ushort 0x003 ushort 0x004 ulong 0x00aa end",
        "res refused replay end",
        "rep refused replay end"]),
    ("mm_transport_then_verdict", "mm", (0x010, 0x020, 0x030), (0x0bad,), [[T_, ("fail", "verdict")]], [
        "goal mm ushort 0x010 ushort 0x020 ushort 0x030 ulong 0x0bad end",
        "act job mm ushort 0x010 ushort 0x020 ushort 0x030 ulong 0x0bad end",
        "res fail transport end",
        "act job mm ushort 0x010 ushort 0x020 ushort 0x030 ulong 0x0bad end",
        "res fail verdict end",
        "rep fail verdict end"]),
    ("pair_ok_then_transport2", "pair", (0x008, 0x008, 0x008), (0x0001, 0x0002), [[OK], [T_, T_]], [
        "goal pair ushort 0x008 ushort 0x008 ushort 0x008 ulong 0x0001 ulong 0x0002 end",
        "act job mm ushort 0x008 ushort 0x008 ushort 0x008 ulong 0x0001 end",
        "res ok end",
        "act job mm ushort 0x008 ushort 0x008 ushort 0x008 ulong 0x0002 end",
        "res fail transport end",
        "act job mm ushort 0x008 ushort 0x008 ushort 0x008 ulong 0x0002 end",
        "res fail transport end",
        "rep fail transport end"]),
    ("pair_budget_after_retry", "pair", (0x00c, 0x00d, 0x00e), (0x0100, 0x0200), [[T_, ("refused", "budget")]], [
        "goal pair ushort 0x00c ushort 0x00d ushort 0x00e ulong 0x0100 ulong 0x0200 end",
        "act job mm ushort 0x00c ushort 0x00d ushort 0x00e ulong 0x0100 end",
        "res fail transport end",
        "act job mm ushort 0x00c ushort 0x00d ushort 0x00e ulong 0x0100 end",
        "res refused budget end",
        "rep refused budget end"]),
    ("pair_max_175", "pair", (0x0ff, 0x0ff, 0x0ff), (0xbeef, 0xcafe), [[T_, OK], [T_, OK]], [
        "goal pair ushort 0x0ff ushort 0x0ff ushort 0x0ff ulong 0xbeef ulong 0xcafe end",
        "act job mm ushort 0x0ff ushort 0x0ff ushort 0x0ff ulong 0xbeef end",
        "res fail transport end",
        "act job mm ushort 0x0ff ushort 0x0ff ushort 0x0ff ulong 0xbeef end",
        "res ok end",
        "act job mm ushort 0x0ff ushort 0x0ff ushort 0x0ff ulong 0xcafe end",
        "res fail transport end",
        "act job mm ushort 0x0ff ushort 0x0ff ushort 0x0ff ulong 0xcafe end",
        "res ok end",
        "rep ok pair ushort 0x0ff ushort 0x0ff ushort 0x0ff ulong 0xbeef ulong 0xcafe end"]),
]

# Hand-written violations: (name, lines) each must fail the audit.
G1 = "goal mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1234 end"
A1 = "act job mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1234 end"
GP = "goal pair ushort 0x008 ushort 0x008 ushort 0x008 ulong 0x0001 ulong 0x0002 end"
AP1 = "act job mm ushort 0x008 ushort 0x008 ushort 0x008 ulong 0x0001 end"
AP2 = "act job mm ushort 0x008 ushort 0x008 ushort 0x008 ulong 0x0002 end"
BAD = [
    ("no_retry_after_first_transport", [G1, A1, "res fail transport end", "rep fail transport end"]),
    ("third_attempt", [G1, A1, "res fail transport end", A1, "res fail transport end", A1, "res ok end",
                       "rep ok mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1234 end"]),
    ("pair_job2_wrong_seed", [GP, AP1, "res ok end", AP1, "res ok end",
                              "rep ok pair ushort 0x008 ushort 0x008 ushort 0x008 ulong 0x0001 ulong 0x0002 end"]),
    ("refuse_admitted_0x100", ["goal mm ushort 0x100 ushort 0x004 ushort 0x004 ulong 0x0001 end",
                               "act refuse bound end", "rep refused bound end"]),
    ("dispatch_0x101", ["goal mm ushort 0x101 ushort 0x004 ushort 0x004 ulong 0x0001 end",
                        "act job mm ushort 0x101 ushort 0x004 ushort 0x004 ulong 0x0001 end", "res ok end",
                        "rep ok mm ushort 0x101 ushort 0x004 ushort 0x004 ulong 0x0001 end"]),
    ("report_wrong_code", [G1, A1, "res refused budget end", "rep refused worker end"]),
    ("report_wrong_class", [G1, A1, "res fail io end", "rep refused io end"]),
    ("rep_ok_wrong_seed", [G1, A1, "res ok end",
                           "rep ok mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1235 end"]),
    ("res_after_refuse", ["goal mm ushort 0x101 ushort 0x004 ushort 0x004 ulong 0x0001 end",
                          "act refuse bound end", "res ok end", "rep refused bound end"]),
    ("res_bound_runtime", [G1, A1, "res refused bound end", "rep refused bound end"]),
    ("missing_report", [G1, A1, "res ok end"]),
]
BAD_SURFACE = ["goal mm ushort 0X040 ushort 0x040 ushort 0x040 ulong 0x1234 end",
               "goal mm ushort 0x040  ushort 0x040 ushort 0x040 ulong 0x1234 end",
               "Goal mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1234 end",
               "goal mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1234",
               "goal mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1234 end end",
               "goal mm ushort 0x40 ushort 0x040 ushort 0x040 ulong 0x1234 end",
               "goal mm ushort 0x040\tushort 0x040 ushort 0x040 ulong 0x1234 end",
               "res ok end\r", ""]


def spec_tests():
    for name, kind, dims, seeds, script, lines in SPEC:
        got = oe_sealgen.build_episode(kind, dims, seeds, script)
        t("spec.%s.generator" % name, got == lines, "" if got == lines else repr(got))
        ok, why, info = oe_audit.audit_one(lines)
        t("spec.%s.audit" % name, ok, why)
    n = L.episode_ids_len(SPEC[-1][5])
    t("spec.maxlen_episode_is_175_ids", n == 175, "ids=%d" % n)
    for name, lines in BAD:
        ok, why, _ = oe_audit.audit_one(lines)
        t("neg.%s.rejected" % name, not ok, why)
    for s in BAD_SURFACE:
        try:
            L.parse_line(s)
            rejected = False
        except ValueError:  # CodecError is a ValueError; grammar errors too
            rejected = True
        t("neg.surface.%r" % s[:24], rejected)


def contract_vocab_test(path):
    if not path:
        print("T contract.vocab_tsv NOT_RUN no vocab.tsv path given")
        return
    rows = open(path, newline="").read().split("\n")
    words = [r.split("\t")[1] for r in rows if r]
    ids = [int(r.split("\t")[0]) for r in rows if r]
    t("contract.vocab_tsv_matches_eval_vocab", ids == list(range(41)) and words == L.VOCAB,
      "sha256=%s" % hashlib.sha256(open(path, "rb").read()).hexdigest())


# ------------------------------------------------------- 2. sealgen determinism
def sealgen_tests(tmp):
    seed = os.path.join(tmp, "seed.hex")
    open(seed, "w").write("00" * 31 + "01\n")
    outs = []
    for i in range(2):
        o = os.path.join(tmp, "s%d" % i)
        subprocess.run([sys.executable, os.path.join(EVAL, "oe_sealgen.py"), seed, o], check=True,
                       stdout=subprocess.DEVNULL)
        outs.append(open(o, "rb").read())
    t("sealgen.deterministic", outs[0] == outs[1], hashlib.sha256(outs[0]).hexdigest())
    r = subprocess.run([sys.executable, os.path.join(EVAL, "oe_audit.py"), "bundle", os.path.join(tmp, "s0")],
                       capture_output=True, text=True)
    t("sealgen.audit_pass", r.returncode == 0 and "AUDIT PASS" in r.stdout, r.stdout.strip().splitlines()[-1])


# ----------------------------------------------------- 3. real artifacts
BASE = os.path.join(HERE, "fixtures", "baseline")
DAG = "aadc3775d32303b9e697c1c092b5628e9333f012a2ac4c1390f04da113c494cb"


def keyfile(name):
    return {l.split()[0]: bytes.fromhex(l.split()[1]) for l in open(os.path.join(BASE, name)) if l.strip()}


def real_artifact_tests():
    sk, mk = keyfile("spark-public-keys.txt"), keyfile("macbook-public-keys.txt")
    keys = (sk["submitter.p256"], mk["executor.p256"], sk["verifier.p256"])
    rd = lambda e: open(os.path.join(BASE, DAG + e), "rb").read()
    an0b, mjr1, mjv = rd(".an0b"), rd(".mjr1"), rd(".mjv")
    t("p256.curve_self_test", A.curve_self_test())
    res, job = A.check_triple(an0b, mjr1, mjv, *keys)
    fails = [r for r in res if r[1] != "PASS"]
    t("real.baseline_all_pass", not fails and job == (64, 64, 64, 2716205057), repr(fails))
    t("real.mjr1_sha256", A.sha(mjr1).hex() == "3306d180de1885a2759b324618d11eca523ef4d7aa22c4144813d71187014855")
    status = lambda rs, name: [s for n, s, _ in rs if n == name]
    # negative controls on real bytes
    flip = bytearray(mjr1)
    flip[100] ^= 1
    res, _ = A.check_triple(an0b, bytes(flip), mjv, *keys)
    t("real.neg.receipt_byte_flip", status(res, "mjr1.signature") == ["FAIL"])
    flip = bytearray(mjr1)
    flip[150] ^= 1
    res, _ = A.check_triple(an0b, bytes(flip), mjv, *keys)
    t("real.neg.output_digest_flip", status(res, "recompute.output_digest") == ["FAIL"]
      and status(res, "mjr1.signature") == ["FAIL"])
    hs = bytearray(mjr1)
    s = int.from_bytes(mjr1[372:404], "big")
    hs[372:404] = (A.N - s).to_bytes(32, "big")
    ok, why = A.verify_low_s(mjr1[304:337], mjr1[:340], bytes(hs[340:404]))
    t("real.neg.high_s_refused", not ok and why == "high-S signature", why)
    t("real.crosscheck_high_s_valid_ecdsa", A.crypto_crosscheck(mjr1[304:337], mjr1[:340], bytes(hs[340:404])) in (True, None))
    ok, _ = A.verify_low_s(keys[2], A.VERIFY_DOMAIN + mjv[:172], mjv[172:236])
    t("real.an0v_signature", ok)
    res, _ = A.check_triple(an0b, mjr1, mjv, keys[0], keys[2], keys[2])
    t("real.neg.wrong_executor_pin", status(res, "mjr1.executor_pin") == ["FAIL"])
    t("matmul.distinct_seed_changes_digest",
      A.matmul_u32_digest(64, 64, 64, 2716205058) != mjr1[144:176])


# ----------------------------------------------------- 4. check-demo
CHECK = os.path.join(EVAL, "check-demo")


def run_check(case):
    r = subprocess.run([sys.executable, CHECK, case], capture_output=True, text=True)
    return r.returncode, r.stdout


def mm_case(path, forge_output=False):
    lines = SPEC[0][5]
    mkcase.write_case(path, lines, (0x40, 0x40, 0x40), (0x1234,), [(1, 1, "ok", None)],
                      forge_output=forge_output)
    rows = open(os.path.join(path, "facts.tsv")).read().splitlines()[1].split("\t")
    eng = ("Completed and verified: one matmul_u32 job multiplying a 64 by 64 matrix by a 64 by 64 matrix "
           "with seed 4660; job 1 dag_id %s mjr1_sha256 %s.\n" % (rows[9], rows[10]))
    open(os.path.join(path, "response.en"), "w").write(eng)
    return eng


def checkdemo_tests(tmp):
    happy = os.path.join(tmp, "happy-mm")
    eng = mm_case(happy)
    rc, out = run_check(happy)
    t("checkdemo.happy_mm", rc == 0 and out.endswith("FAITHFUL\n"), out.splitlines()[-1])

    pair = os.path.join(tmp, "happy-pair")
    lines = SPEC[8][5]
    mkcase.write_case(pair, lines, (0xff,) * 3, (0xbeef, 0xcafe),
                      [(1, 1, "fail", "transport"), (1, 2, "ok", None), (2, 1, "fail", "transport"), (2, 2, "ok", None)])
    rows = [r.split("\t") for r in open(os.path.join(pair, "facts.tsv")).read().splitlines()[1:]]
    open(os.path.join(pair, "response.en"), "w").write(
        "Completed and verified: two matmul_u32 jobs, each multiplying a 255 by 255 matrix by a 255 by 255 matrix, "
        "with seeds 48879 and 51966; job 1 dag_id %s mjr1_sha256 %s; job 2 dag_id %s mjr1_sha256 %s.\n"
        % (rows[1][9], rows[1][10], rows[3][9], rows[3][10]))
    rc, out = run_check(pair)
    t("checkdemo.pair_with_retries", rc == 0, out.splitlines()[-1])

    bound = os.path.join(tmp, "bound")
    mkcase.write_case(bound, SPEC[2][5], (0x101, 4, 4), (7,), [],
                      response="Not completed: refused before dispatch (code bound): the goal asks for a 257 by 4 by 4 "
                               "matmul and a dimension is outside the admitted range 1 to 256; nothing was dispatched.\n")
    rc, out = run_check(bound)
    t("checkdemo.bound", rc == 0, out.splitlines()[-1])

    stop = os.path.join(tmp, "replay")
    mkcase.write_case(stop, SPEC[4][5], (2, 3, 4), (0xaa,), [(1, 1, "refused", "replay")],
                      response="Not completed: job 1 was refused (code replay): the worker refused a replay of an "
                               "already executed job.\n")
    rc, out = run_check(stop)
    t("checkdemo.replay", rc == 0, out.splitlines()[-1])

    tr = os.path.join(tmp, "translator-refusal")
    os.makedirs(os.path.join(tr, "artifacts"))
    open(os.path.join(tr, "facts.tsv"), "w").write(mkcase.HEADER + "\n")
    open(os.path.join(tr, "request.en"), "w").write("please sort my files\n")
    rc, out = run_check(tr)
    t("checkdemo.translator_refusal", rc == 0, out.splitlines()[-1])

    def neg(name, mutate, expect):
        d = os.path.join(tmp, "neg-" + name)
        shutil.copytree(happy, d)
        mutate(d)
        rc, out = run_check(d)
        last = out.splitlines()[-1] if out else ""
        t("checkdemo.neg.%s" % name, rc == 1 and last.startswith("UNFAITHFUL") and expect in out, last)

    def art(d, ext):
        a = os.path.join(d, "artifacts")
        return os.path.join(a, [f for f in os.listdir(a) if f.endswith(ext)][0])

    def flip(path, off):
        b = bytearray(open(path, "rb").read())
        b[off] ^= 1
        open(path, "wb").write(bytes(b))

    neg("receipt_byte", lambda d: flip(art(d, ".mjr1"), 90), "mjr1.signature FAIL")
    # Re-signed forgery: the worker's digest is wrong by one bit, but receipt, verdict
    # commitment, countersignature, facts.tsv and the English sha are all consistent
    # and validly signed. The ONLY failing check may be the independent recompute.
    forged = os.path.join(tmp, "neg-output_digest_resigned")
    feng = mm_case(forged, forge_output=True)
    rc, out = run_check(forged)
    failed = [l.split()[1] for l in out.splitlines() if l.startswith("CHECK ") and l.split()[2] == "FAIL"]
    t("checkdemo.neg.output_digest_resigned", rc == 1 and failed == ["row1.recompute.output_digest"]
      and out.splitlines()[-1].startswith("UNFAITHFUL row1.recompute.output_digest"), "failed=%r" % failed)
    t("checkdemo.forged_differs_from_honest_only_downstream_of_digest",
      feng != eng and feng.split("dag_id ")[1][:64] == eng.split("dag_id ")[1][:64])
    neg("missing_response", lambda d: os.remove(os.path.join(d, "response.en")), "english.present FAIL")
    neg("english_number", lambda d: open(os.path.join(d, "response.en"), "w").write(
        eng.replace("with seed 4660", "with seed 4661")), "english.seeds FAIL")
    neg("english_dim", lambda d: open(os.path.join(d, "response.en"), "w").write(
        eng.replace("a 64 by 64 matrix by", "a 65 by 64 matrix by")), "english.dims FAIL")
    neg("english_unparsed", lambda d: open(os.path.join(d, "response.en"), "w").write(
        eng.replace("Completed and verified", "Completed")), "english.parse FAIL")
    neg("countersign_byte", lambda d: flip(art(d, ".mjv"), 200), "an0v.signature FAIL")

    def wrong_seed_row(d):
        p = os.path.join(d, "facts.tsv")
        s = open(p).read().replace("\t4660\t", "\t4661\t")
        open(p, "w").write(s)
    neg("facts_seed", wrong_seed_row, "binds_goal FAIL")

    def unbound(d):
        p = os.path.join(d, "episode.oe45")
        s = open(p).read().replace("act job mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1234",
                                   "act job mm ushort 0x040 ushort 0x040 ushort 0x040 ulong 0x1235")
        open(p, "w").write(s)
    neg("unbound_action", unbound, "episode.binding FAIL")

    def wrong_reason(d):
        mkcase.write_case(d, SPEC[4][5], (2, 3, 4), (0xaa,), [(1, 1, "refused", "replay")],
                          response="Not completed: job 1 was refused (code replay): the worker refused the job.\n")
        for f in os.listdir(os.path.join(d, "artifacts")):
            os.remove(os.path.join(d, "artifacts", f))
    neg("english_reason", wrong_reason, "english.reason FAIL")

    # interface v1.2 facts.tsv rules
    def set_field(d, rowno, col, value, append_copy=False):
        p = os.path.join(d, "facts.tsv")
        lines = open(p).read().splitlines()
        f = lines[rowno].split("\t")
        f[mkcase.HEADER.split("\t").index(col)] = value
        if append_copy:
            lines.append("\t".join(f))
        else:
            lines[rowno] = "\t".join(f)
        open(p, "w").write("\n".join(lines) + "\n")

    def replay_case(d):
        mkcase.write_case(d, SPEC[4][5], (2, 3, 4), (0xaa,), [(1, 1, "refused", "replay")],
                          response="Not completed: job 1 was refused (code replay): the worker refused a replay "
                                   "of an already executed job.\n")

    neg("facts_outcome_not_1", lambda d: set_field(d, 1, "mjr1_outcome", "2"), "row1.facts_format FAIL")
    neg("facts_verifier_uppercase", lambda d: set_field(d, 1, "verifier", open(
        os.path.join(d, "facts.tsv")).read().splitlines()[1].split("\t")[12].upper()), "row1.facts_format FAIL")
    neg("facts_verifier_not_compressed", lambda d: set_field(d, 1, "verifier", "04" + "ab" * 32),
        "row1.facts_format FAIL")
    neg("facts_row_after_ok", lambda d: set_field(d, 1, "attempt", "2", append_copy=True),
        "facts.no_row_after_ok FAIL")
    neg("facts_attempt_3", lambda d: set_field(d, 1, "attempt", "3"), "row1.facts_format FAIL")
    neg("facts_exit_map", lambda d: (replay_case(d), set_field(d, 1, "exit", "12")), "row1.facts_format FAIL")
    neg("facts_nonok_receipt_field", lambda d: (replay_case(d), set_field(d, 1, "verifier", "02" + "ab" * 32)),
        "row1.facts_format FAIL")
    # parent ruling on v1.2: a non-ok row may carry the real dag_id ('-' or 64 lowercase hex)
    DEMO_DAG = "32d3147d8786f0f3be217de5dbd512191229f7dd80e89f4d9bc92f0200b97fca"  # demo replay row
    pos = os.path.join(tmp, "replay-with-dag")
    replay_case(pos)
    set_field(pos, 1, "dag_id", DEMO_DAG)
    rc, out = run_check(pos)
    t("checkdemo.nonok_row_with_dag_id_accepted", rc == 0 and out.endswith("FAITHFUL\n"), out.splitlines()[-1])
    neg("facts_nonok_dag_nonhex", lambda d: (replay_case(d), set_field(d, 1, "dag_id", "g" + DEMO_DAG[1:])),
        "row1.facts_format FAIL non-ok row dag_id")
    neg("facts_nonok_dag_short", lambda d: (replay_case(d), set_field(d, 1, "dag_id", DEMO_DAG[:63])),
        "row1.facts_format FAIL non-ok row dag_id")
    neg("facts_nonok_dag_upper", lambda d: (replay_case(d), set_field(d, 1, "dag_id", DEMO_DAG.upper())),
        "row1.facts_format FAIL non-ok row dag_id")
    neg("facts_nonok_mjr1_without_receipt",
        lambda d: (replay_case(d), set_field(d, 1, "dag_id", DEMO_DAG), set_field(d, 1, "mjr1_sha256", "ab" * 32)),
        "row1.nonok_receipt_file FAIL")
    rc = subprocess.run([sys.executable, CHECK], capture_output=True).returncode
    t("checkdemo.usage_rc2", rc == 2)
    return happy


def main(argv):
    with tempfile.TemporaryDirectory(prefix="oe45-eval-test-") as tmp:
        spec_tests()
        contract_vocab_test(argv[1] if len(argv) > 1 else None)
        sealgen_tests(tmp)
        real_artifact_tests()
        checkdemo_tests(tmp)
    print("OE45_EVAL_TESTS %s %s" % ("FAIL" if FAILS else "PASS", " ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
