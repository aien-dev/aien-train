"""Eval-owned readers for the Aienet-0 job path artifacts (aien-train #45).

Written from aien-protocols e705fdc9, not from any #45 lane's code:
  specs/microjob/SPEC.md sections 4 and 5 (MJB1 document, MJR1 receipt),
  specs/aienet/jobpath/PROFILE.md (AN0B bundle, AN0V countersignature,
  matmul_u32 payload and digest), crates/aien-microjob/src/{lib.rs,aienet0.rs}
  (byte offsets, domains), crates/aien-provenance/src/lib.rs:215-248
  (ECDSA P-256 / SHA-256, raw r||s, low-S only, SEC1 compressed keys).

Python standard library only. P-256 verification is implemented here from the
SEC 1 / FIPS 186 definitions; tests cross-check it against real Rust-signed
artifacts and, when importable, the `cryptography` package.
"""

import hashlib
import struct

# ---------------------------------------------------------------- P-256
P = 2**256 - 2**224 + 2**192 + 2**96 - 1
A = P - 3
B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551


def _on_curve(pt):
    if pt is None:
        return True
    x, y = pt
    return (y * y - (x * x * x + A * x + B)) % P == 0


def _add(p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None
        lam = (3 * x1 * x1 + A) * pow(2 * y1, -1, P) % P
    else:
        lam = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    return (x3, (lam * (x1 - x3) - y1) % P)


def _mul(k, pt):
    acc = None
    while k:
        if k & 1:
            acc = _add(acc, pt)
        pt = _add(pt, pt)
        k >>= 1
    return acc


G = (GX, GY)


def curve_self_test():
    """G on the curve and of order N (catches a mistyped constant)."""
    return _on_curve(G) and _mul(N, G) is None and _mul(N - 1, G) == (GX, P - GY)


def decompress(key33):
    if len(key33) != 33 or key33[0] not in (2, 3):
        raise ValueError("public key is not SEC1 compressed")
    x = int.from_bytes(key33[1:], "big")
    if x >= P:
        raise ValueError("x out of range")
    y2 = (x * x * x + A * x + B) % P
    y = pow(y2, (P + 1) // 4, P)  # P = 3 mod 4
    if y * y % P != y2:
        raise ValueError("x not on curve")
    if (y & 1) != (key33[0] & 1):
        y = P - y
    return (x, y)


def verify_low_s(key33, message, sig64):
    """(ok, reason). Mirrors p256_verify_low_s: high-S is refused."""
    try:
        q = decompress(bytes(key33))
    except ValueError as e:
        return False, "bad key: %s" % e
    if len(sig64) != 64:
        return False, "signature length"
    r = int.from_bytes(sig64[:32], "big")
    s = int.from_bytes(sig64[32:], "big")
    if not (1 <= r < N and 1 <= s < N):
        return False, "r or s out of range"
    if s > N // 2:
        return False, "high-S signature"
    e = int.from_bytes(hashlib.sha256(message).digest(), "big")
    w = pow(s, -1, N)
    pt = _add(_mul(e * w % N, G), _mul(r * w % N, q))
    if pt is None:
        return False, "point at infinity"
    if pt[0] % N != r:
        return False, "signature does not verify"
    return True, "ok"


def crypto_crosscheck(key33, message, sig64):
    """Optional cross-check with the `cryptography` package: True/False, or
    None when the package is not importable."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
    except ImportError:
        return None
    try:
        pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), bytes(key33))
        der = encode_dss_signature(int.from_bytes(sig64[:32], "big"), int.from_bytes(sig64[32:], "big"))
        pub.verify(der, bytes(message), ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError):
        return False


# ---------------------------------------------------------------- matmul_u32
MATMUL_DOMAIN = b"AIEN-HETJOB-MATMUL-U32-V1"
MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1


def matmul_u32_digest(m, k, n, seed):
    """PROFILE.md 'The job': A (m x k) then B (k x n) from splitmix64(seed),
    each truncated to u32; C = A x B mod 2^32; SHA-256(domain || m || k || n
    || seed || C as u32 BE row-major)."""
    s = seed & MASK64
    vals = []
    for _ in range(m * k + k * n):
        s = (s + 0x9E3779B97F4A7C15) & MASK64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        vals.append((z ^ (z >> 31)) & MASK32)
    a = vals[: m * k]
    b = vals[m * k:]
    cols = [b[q::n] for q in range(n)]
    h = hashlib.sha256()
    h.update(MATMUL_DOMAIN + struct.pack(">HHHQ", m, k, n, seed & MASK64))
    out = bytearray()
    for i in range(m):
        row = a[i * k:(i + 1) * k]
        for col in cols:
            out += (sum(map(int.__mul__, row, col)) & MASK32).to_bytes(4, "big")
    h.update(bytes(out))
    return h.digest()


# ---------------------------------------------------------------- records
def sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


class Bad(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Bad(msg)


BUNDLE_HEAD = 4 + 1 + 1 + 2 + 4 + 33 + 64
SUBMIT_DOMAIN = b"AIENET0-MJB1-SUBMIT-V1\x00"
VERIFY_DOMAIN = b"AIENET0-MJR1-VERIFY-V1\x00"
VERDICT_DOMAIN = b"AIENET0-RECOMPUTE-MATCH-V1\x00"
HEADER_LEN = 192
NODE_LEN = 448
RECEIPT_LEN = 404
RECEIPT_SIGNED_LEN = 340
VERDICT_SIGNED_LEN = 172
VERDICT_LEN = 236


def parse_bundle(b):
    need(len(b) >= BUNDLE_HEAD and b[:4] == b"AN0B", "AN0B magic/length")
    need(b[4] == 1 and b[5] == 0, "AN0B version/reserved")
    plen, dlen = struct.unpack(">HI", b[6:12])
    need(plen <= 256 and BUNDLE_HEAD + plen + dlen == len(b), "AN0B lengths")
    payload = b[BUNDLE_HEAD:BUNDLE_HEAD + plen]
    return {"key": b[12:45], "sig": b[45:109], "payload": payload, "document": b[BUNDLE_HEAD + plen:]}


def parse_payload(p):
    need(len(p) == 15 and p[0] == 1, "payload is not matmul_u32 (kind 1, 15 bytes)")
    m, k, n, seed = struct.unpack(">HHHQ", p[1:15])
    return m, k, n, seed


def parse_document(d):
    """The fields of SPEC.md 4.1/4.2 that bind a profile document to its job."""
    need(len(d) >= HEADER_LEN and d[:4] == b"MJB1" and d[4] == 1, "MJB1 magic/version")
    (count,) = struct.unpack(">H", d[6:8])
    need(len(d) == HEADER_LEN + count * NODE_LEN, "MJB1 length")
    need(count == 1, "profile document has exactly one node")
    nd = d[HEADER_LEN:HEADER_LEN + NODE_LEN]
    ops = []
    for s in range(nd[55]):
        at = 288 + 40 * s
        ops.append({"kind": nd[at], "generation": struct.unpack(">I", nd[at + 4:at + 8])[0], "digest": nd[at + 8:at + 40]})
    return {
        "expires_at": struct.unpack(">Q", d[144:152])[0],
        "job_id": nd[0:16],
        "idem": nd[16:32],
        "generation": struct.unpack(">I", nd[32:36])[0],
        "kind": nd[36],
        "op": nd[37],
        "verifier_kind": nd[48],
        "dep_count": nd[54],
        "work": struct.unpack(">Q", nd[80:88])[0],
        "operands": ops,
    }


def parse_receipt(r):
    need(len(r) == RECEIPT_LEN and r[:4] == b"MJR1" and r[4] == 1, "MJR1 length/magic/version")
    need(r[7] == 0 and r[84:88] == b"\0" * 4 and r[337:340] == b"\0" * 3, "MJR1 reserved bytes")
    need(1 <= r[5] <= 5 and r[6] != 0, "MJR1 outcome/attempt")
    need(r[304] in (2, 3), "MJR1 signer key prefix")
    completed = r[5] == 1
    zero = lambda x: x == b"\0" * len(x)
    need(completed != zero(r[144:176]), "MJR1 output_digest zero rule")
    need(not (completed and zero(r[272:304])), "MJR1 verifier_verdict zero rule")
    need(not zero(r[176:208]) and not zero(r[240:272]), "MJR1 source/policy hash zero")
    return {
        "outcome": r[5],
        "attempt": r[6],
        "dag_digest": r[8:40],
        "job_id": r[40:56],
        "idem": r[56:72],
        "generation": struct.unpack(">I", r[72:76])[0],
        "used_work": struct.unpack(">Q", r[104:112])[0],
        "input_binding": r[112:144],
        "output_digest": r[144:176],
        "verdict": r[272:304],
        "signer": r[304:337],
        "sig": r[340:404],
    }


def parse_countersign(v):
    need(len(v) == VERDICT_LEN and v[:4] == b"AN0V" and v[4] == 1, "AN0V length/magic/version")
    need(v[5:8] == b"\0\0\0" and v[169:172] == b"\0\0\0", "AN0V reserved bytes")
    return {
        "receipt_sha": v[8:40],
        "dag_digest": v[40:72],
        "recomputed": v[72:104],
        "commit": v[104:136],
        "key": v[136:169],
        "sig": v[172:236],
    }


def input_binding(operands):
    h = hashlib.sha256(b"MJR1-BIND")
    for o in operands:
        h.update(bytes([o["kind"]]) + struct.pack(">I", o["generation"]) + o["digest"])
    return h.digest()


def check_triple(an0b, mjr1, mjv, submitter, executor, verifier, crosscheck=True):
    """Independent re-check of one stored (bundle, receipt, countersignature).

    Returns a list of (name, status, detail) with status PASS/FAIL/NOT_RUN and
    the job (m, k, n, seed) or None. `mjv` may be None (no countersignature
    written, e.g. a failed verification); then the AN0V checks are NOT_RUN.
    """
    out = []

    def rec(name, ok, detail=""):
        out.append((name, "PASS" if ok else "FAIL", detail))
        return ok

    job = None
    try:
        bd = parse_bundle(an0b)
        job = parse_payload(bd["payload"])
        doc = parse_document(bd["document"])
    except Bad as e:
        rec("an0b.parse", False, str(e))
        return out, None
    rec("an0b.parse", True, "m=%d k=%d n=%d seed=%d" % job)
    dag = sha(bd["document"])
    rec("an0b.submitter_pin", bd["key"] == submitter, bd["key"].hex())
    ok, why = verify_low_s(bd["key"], SUBMIT_DOMAIN + dag + sha(bd["payload"]), bd["sig"])
    rec("an0b.submitter_sig", ok, why)
    rec("mjb1.allowlisted_shape", doc["kind"] == 1 and doc["op"] == 2 and doc["verifier_kind"] == 1
        and doc["dep_count"] == 0 and len(doc["operands"]) == 1, "kind=%d op=%d" % (doc["kind"], doc["op"]))
    rec("mjb1.payload_binding", len(doc["operands"]) == 1 and doc["operands"][0]["kind"] == 3
        and doc["operands"][0]["digest"] == sha(bd["payload"]))  # operand::CONTENT = 3 (lib.rs:98)
    m, k, n, seed = job
    rec("mjb1.work_cap", doc["work"] >= m * k * n, "work=%d" % doc["work"])
    if mjr1 is None:
        out.append(("mjr1.present", "FAIL", "no receipt file"))
        return out, job
    try:
        r = parse_receipt(mjr1)
    except Bad as e:
        rec("mjr1.parse", False, str(e))
        return out, job
    rec("mjr1.parse", True, "outcome=%d attempt=%d" % (r["outcome"], r["attempt"]))
    rec("mjr1.dag_binding", r["dag_digest"] == dag, dag.hex())
    rec("mjr1.node_binding", r["job_id"] == doc["job_id"] and r["idem"] == doc["idem"]
        and r["generation"] == doc["generation"])
    rec("mjr1.input_binding", r["input_binding"] == input_binding(doc["operands"]))
    rec("mjr1.executor_pin", r["signer"] == executor, r["signer"].hex())
    ok, why = verify_low_s(r["signer"], mjr1[:RECEIPT_SIGNED_LEN], r["sig"])
    rec("mjr1.signature", ok, why)
    if crosscheck:
        cc = crypto_crosscheck(r["signer"], mjr1[:RECEIPT_SIGNED_LEN], r["sig"])
        if cc is None:
            out.append(("mjr1.signature_crosscheck", "NOT_RUN", "cryptography not importable"))
        else:
            rec("mjr1.signature_crosscheck", cc == ok, "cryptography=%s" % cc)
    rec("mjr1.completed", r["outcome"] == 1, "outcome=%d" % r["outcome"])
    want = matmul_u32_digest(m, k, n, seed)
    rec("recompute.output_digest", r["output_digest"] == want, "eval=%s" % want.hex())
    commit = sha(VERDICT_DOMAIN, dag, r["job_id"], want)
    rec("mjr1.verdict_commitment", r["verdict"] == commit)
    if mjv is None:
        out.append(("an0v.present", "NOT_RUN", "no countersignature file"))
        return out, job
    try:
        v = parse_countersign(mjv)
    except Bad as e:
        rec("an0v.parse", False, str(e))
        return out, job
    rec("an0v.receipt_binding", v["receipt_sha"] == sha(mjr1) and v["dag_digest"] == dag)
    rec("an0v.recompute_binding", v["recomputed"] == want and v["commit"] == commit)
    rec("an0v.verifier_pin", v["key"] == verifier, v["key"].hex())
    ok, why = verify_low_s(v["key"], VERIFY_DOMAIN + mjv[:VERDICT_SIGNED_LEN], v["sig"])
    rec("an0v.signature", ok, why)
    return out, job
