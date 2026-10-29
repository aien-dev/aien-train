"""Test-only builder of synthetic demo case directories for check-demo.

Signs with throwaway test keys (private scalars below are public test data,
never used anywhere else). Byte layouts follow aien-protocols e705fdc9
specs/microjob/SPEC.md and specs/aienet/jobpath/PROFILE.md. The layout
readers in eval/omegese45/oe_an0.py are proven against the real Rust-signed
artifacts in fixtures/baseline/, so these fixtures only exercise check-demo's
plumbing (episode <-> facts <-> artifacts <-> English) and its negative
controls.
"""

import hashlib
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "eval", "omegese45"))
import oe_an0 as A  # noqa: E402

TEST_KEYS = {"submitter": 0x5EED01, "executor": 0x5EED02, "verifier": 0x5EED03}


def pub(d):
    x, y = A._mul(d, A.G)
    return bytes([2 | (y & 1)]) + x.to_bytes(32, "big")


def sign(d, msg):
    e = int.from_bytes(hashlib.sha256(msg).digest(), "big")
    k = int.from_bytes(hashlib.sha256(d.to_bytes(32, "big") + msg).digest(), "big") % A.N or 1
    r = A._mul(k, A.G)[0] % A.N
    s = pow(k, -1, A.N) * (e + r * d) % A.N
    if s > A.N // 2:
        s = A.N - s
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def document(payload, job_id, m, k, n):
    head = bytearray(192)
    head[0:4] = b"MJB1"
    head[4] = 1
    head[6:8] = struct.pack(">H", 1)
    head[144:152] = struct.pack(">Q", 4102444800)
    node = bytearray(448)
    node[0:16] = job_id
    node[16:32] = hashlib.sha256(b"idem" + job_id).digest()[:16]
    node[32:36] = struct.pack(">I", 1)
    node[36], node[37], node[48], node[54], node[55] = 1, 2, 1, 0, 1
    node[80:88] = struct.pack(">Q", m * k * n)
    node[288] = 3
    node[292:296] = struct.pack(">I", 1)
    node[296:328] = hashlib.sha256(payload).digest()
    return bytes(head + node)


def triple(m, k, n, seed, tag=b"", forge_output=False):
    """forge_output: the worker reports a wrong output digest (one bit flipped)
    and every party signs consistently over it: the receipt, the verdict
    commitment and the countersignature are all valid, so only an independent
    recompute can reject it."""
    payload = bytes([1]) + struct.pack(">HHHQ", m, k, n, seed)
    job_id = hashlib.sha256(b"job" + payload + tag).digest()[:16]
    doc = document(payload, job_id, m, k, n)
    dag = A.sha(doc)
    sk = TEST_KEYS
    sig = sign(sk["submitter"], A.SUBMIT_DOMAIN + dag + A.sha(payload))
    an0b = (b"AN0B" + bytes([1, 0]) + struct.pack(">HI", len(payload), len(doc))
            + pub(sk["submitter"]) + sig + payload + doc)
    out = A.matmul_u32_digest(m, k, n, seed)
    if forge_output:
        out = bytes([out[0] ^ 1]) + out[1:]
    commit = A.sha(A.VERDICT_DOMAIN, dag, job_id, out)
    ops = A.parse_document(doc)["operands"]
    r = bytearray(404)
    r[0:4] = b"MJR1"
    r[4], r[5], r[6] = 1, 1, 1
    r[8:40] = dag
    r[40:56] = job_id
    r[56:72] = doc[192 + 16:192 + 32]
    r[72:76] = struct.pack(">I", 1)
    r[88:112] = struct.pack(">QQQ", m * k * n, 4 * (m * k + k * n + m * n), m * k * n)
    r[112:144] = A.input_binding(ops)
    r[144:176] = out
    r[176:208] = hashlib.sha256(b"source").digest()
    r[240:272] = hashlib.sha256(b"policy").digest()
    r[272:304] = commit
    r[304:337] = pub(sk["executor"])
    r[340:404] = sign(sk["executor"], bytes(r[:340]))
    mjr1 = bytes(r)
    v = bytearray(236)
    v[0:4] = b"AN0V"
    v[4] = 1
    v[8:40] = A.sha(mjr1)
    v[40:72] = dag
    v[72:104] = out
    v[104:136] = commit
    v[136:169] = pub(sk["verifier"])
    v[172:236] = sign(sk["verifier"], A.VERIFY_DOMAIN + bytes(v[:172]))
    return dag.hex(), an0b, mjr1, bytes(v)


HEADER = "job\tattempt\tm\tk\tn\tseed\texit\tclass\tcode\tdag_id\tmjr1_sha256\tmjr1_outcome\tverifier"
CODE_EXIT = {"transport": 10, "replay": 11, "budget": 11, "worker": 11, "receipt": 12,
             "recomp": 13, "verdict": 14, "io": 15}


def write_case(path, lines, dims, seeds, outcomes, response=None, request="request\n", forge_output=False):
    """lines: episode message lines; outcomes: per dispatch (job, attempt, cls, code)."""
    os.makedirs(os.path.join(path, "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(path, "keys"), exist_ok=True)
    os.makedirs(os.path.join(path, "pins"), exist_ok=True)
    os.makedirs(os.path.join(path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(path, "worker"), exist_ok=True)
    sk = TEST_KEYS
    with open(os.path.join(path, "keys", "coordinator-public-keys.txt"), "w") as f:
        f.write("submitter.p256 %s\nverifier.p256 %s\n" % (pub(sk["submitter"]).hex(), pub(sk["verifier"]).hex()))
        f.write("coordinator.ed25519 897507c5dc043871fc4592c7c723a6bf4d90d94cf56d2db8dc71bf79e283b8c6\n")
    with open(os.path.join(path, "keys", "worker-public-keys.txt"), "w") as f:
        f.write("executor.p256 %s\n" % pub(sk["executor"]).hex())
        f.write("volunteer.ed25519 058df02bc0cf534602497686ecb8e4ae6ae43d4dbb44fbf5fcb1a8c37a2ac1bd\n")
    with open(os.path.join(path, "pins", "executor.p256"), "w") as f:
        f.write(pub(sk["executor"]).hex() + "\n")
    with open(os.path.join(path, "pins", "volunteer.ed25519"), "w") as f:
        f.write("058df02bc0cf534602497686ecb8e4ae6ae43d4dbb44fbf5fcb1a8c37a2ac1bd\n")
    with open(os.path.join(path, "runtime.env"), "w") as f:
        f.write("RECOMPUTE_SHA256=d2cd7382e0dc64e49498ae3b1ceffc73f1ddcbd3ddcf93e6d607498ccd58ea12\n")
    with open(os.path.join(path, "request.en"), "w") as f:
        f.write(request)
    with open(os.path.join(path, "goal.oe45"), "w") as f:
        f.write(lines[0] + "\n")
    with open(os.path.join(path, "episode.oe45"), "w") as f:
        f.write("oe45 v1\n" + "".join(l + "\n" for l in lines))
    rows = [HEADER]
    m, k, n = dims
    dispatch = []
    seen = ["AIENET0-STATE-V1"]
    for ix, (job, attempt, cls, code) in enumerate(outcomes):
        seed = seeds[job - 1]
        if cls == "ok" or code == "replay":
            dag, an0b, mjr1, mjv = triple(m, k, n, seed, b"%d" % attempt, forge_output)
        else:
            # Transport never executes. This test-only log id needs no output
            # recompute or signature; successful rows still get full triples.
            dag = hashlib.sha256(repr((m, k, n, seed, attempt)).encode("ascii")).hexdigest()
        if cls == "ok":
            for ext, b in ((".an0b", an0b), (".mjr1", mjr1), (".mjv", mjv)):
                with open(os.path.join(path, "artifacts", dag + ext), "wb") as f:
                    f.write(b)
            rows.append("\t".join(map(str, [job, attempt, m, k, n, seed, 0, "ok", "-", dag,
                                            A.sha(mjr1).hex(), 1, pub(sk["verifier"]).hex()])))
            err = "aienet0-run: dag_id=%s\naienet0-run: receipt sha256=%s\n" % (dag, A.sha(mjr1).hex())
            exit_code = 0
        else:
            exit_code = CODE_EXIT[code]
            rows.append("\t".join(map(str, [job, attempt, m, k, n, seed, exit_code, cls, code,
                                            dag, "-", "-", "-"])))
            marker = {"transport": "no volunteer connected: fixture",
                      "replay": "REFUSED WORKER AN0_REPLAY",
                      "budget": "REFUSED WORKER AN0_BUDGET", "worker": "REFUSED WORKER",
                      "receipt": "REFUSED AN0_RECEIPT", "recomp": "REFUSED AN0_RECOMPUTE",
                      "verdict": "REFUSED AN0_VERDICT", "io": "REFUSED AN0_OUTPUT_EXISTS"}[code]
            err = "aienet0-run: dag_id=%s\naienet0-run: %s\n" % (dag, marker)
            if code == "replay":
                seen.append("done %s %s %s %s" % (dag, A.sha(mjr1).hex(), pub(sk["executor"]).hex(), mjr1.hex()))
        open(os.path.join(path, "logs", "fixture-%d-run.err" % ix), "w").write(err)
        dispatch.append("outcome %d %d %d %s serve=0" % (job, attempt, exit_code, dag))
    open(os.path.join(path, "dispatch.log"), "w").write("\n".join(dispatch) + "\n")
    open(os.path.join(path, "worker", "seen"), "w").write("\n".join(seen) + "\n")
    with open(os.path.join(path, "facts.tsv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    if response is not None:
        with open(os.path.join(path, "response.en"), "w") as f:
            f.write(response)
