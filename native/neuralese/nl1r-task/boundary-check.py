#!/usr/bin/env python3
"""Compose, compile and exercise the NL1R NL4 boundary adapter (host verification only)."""
import hashlib, json, os, pathlib, subprocess, sys, tempfile

NB = pathlib.Path('/home/drakestapleton/fix/nl-boundary')
TD = pathlib.Path('/home/drakestapleton/fix/nl1r-task/native/neuralese/nl1r-task')
OMC = NB / 'seed/out/omc'
OMC_SHA = 'd6290c7663c2b4bafb59eb3fdf812476e977f4c144def71095c6248ef9b96b84'
BUILD = pathlib.Path('/home/drakestapleton/fix/nl1r-task/build/boundary')
PARTS = [NB / 'build/nl4-deps/h.om', NB / 'neuralese/proposal.om', NB / 'build/nl4-deps/v.om', NB / 'om/tls/sha2.om']
CTX = NB / 'neuralese/fixtures/nl4-context.bin'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def prefix():
    lines = (NB / 'neuralese/host-proposal.om').read_bytes().split(b'\n')
    return b'\n'.join(lines[:144]) + b'\n'


def compose(boundary_src):
    return prefix() + boundary_src + b''.join(p.read_bytes() for p in PARTS)


def compile_src(name, src):
    BUILD.mkdir(parents=True, exist_ok=True)
    s = BUILD / f'{name}.om'
    s.write_bytes(src)
    b = BUILD / name
    with open(s, 'rb') as fi, open(b, 'wb') as fo:
        r = subprocess.run([str(OMC)], stdin=fi, stdout=fo, stderr=subprocess.PIPE, timeout=120)
    out = {'name': name, 'compile_exit': r.returncode, 'compile_stderr': r.stderr.decode(), 'source_sha256': sha(src)}
    if r.returncode == 0:
        os.chmod(b, 0o755)
        out['binary_sha256'] = sha(b.read_bytes())
    return out, b


def run(binp, args, fd=3):
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / 'frame'
        argv = ' '.join(["'%s'" % a for a in [binp] + [str(x) for x in args]])
        cmd = f'exec {argv} {fd}>"{f}" < /dev/null' if fd == 3 else f'exec {argv} < /dev/null'
        r = subprocess.run(['bash', '-c', cmd], capture_output=True, timeout=20)
        data = f.read_bytes() if f.exists() else b''
    return r.returncode, r.stdout.decode(), r.stderr.decode(), data


def expected_stdout(bits, task):
    chunks = 1 if task == 1 else 2
    out = ''
    for c in range(chunks):
        v = (bits >> (2 * c)) & 3
        out += f'NL4R1 0 3 {v} 0 1 1 0 1 0 {v}\n'
    return out


CASES = [
    # name, predicted, target, task, abstain, fd, want_exit
    ('task2-match-6', 6, 6, 2, 0, 3, 0),
    ('task2-match-9', 9, 9, 2, 0, 3, 0),
    ('task2-match-0', 0, 0, 2, 0, 3, 0),
    ('task2-match-15', 15, 15, 2, 0, 3, 0),
    ('task1-match-1', 1, 1, 1, 0, 3, 0),
    ('task1-match-0', 0, 0, 1, 0, 3, 0),
    ('task2-chunk1-wrong', 6, 2, 2, 0, 3, 49),
    ('task2-chunk0-wrong', 6, 4, 2, 0, 3, 49),
    ('task1-mismatch', 1, 0, 1, 0, 3, 49),
    ('task1-oversize-bits', 2, 2, 1, 0, 3, 80),
    ('task-3-refused', 1, 1, 3, 0, 3, 80),
    ('predicted-16-refused', 16, 1, 2, 0, 3, 80),
    ('abstain', 6, 6, 2, 1, 3, 2),
    ('frame-fd-2-refused', 6, 6, 2, 0, 2, 80),
]


def exercise(binp):
    rows = []
    for name, p, t, task, ab, fd, want in CASES:
        rc, so, se, data = run(binp, [CTX, p, t, task, ab, fd], fd=fd if fd == 3 else 2)
        exp_out = expected_stdout(p, task) if want == 0 else ''
        ok = rc == want and (want != 0 or so == exp_out) and (want != 49 or 'NL4R1' not in so)
        rows.append({'case': name, 'args': [str(CTX), p, t, task, ab, fd], 'want_exit': want, 'exit': rc,
                     'stdout': so, 'stderr': se, 'frame_bytes': len(data), 'frame_sha256': sha(data),
                     'verdict': 'PASS' if ok else 'FAIL'})
    rc, so, se, data = run(binp, ['/nonexistent-context', 6, 6, 2, 0, 3])
    rows.append({'case': 'missing-context-refused', 'want_exit': 80, 'exit': rc, 'stdout': so, 'stderr': se,
                 'verdict': 'PASS' if rc == 80 else 'FAIL'})
    rc, so, se, data = run(binp, [CTX, 6, 6, 2, 0])
    rows.append({'case': 'argc-refused', 'want_exit': 80, 'exit': rc, 'stdout': so, 'stderr': se,
                 'verdict': 'PASS' if rc == 80 else 'FAIL'})
    # Envelope negatives: the frame header is copied from the context, so a malformed context must be
    # refused by the unchanged LTP0 validator before any proposal decode or execution.
    for name, mutate, want in BAD_CONTEXTS:
        c = bytearray(CTX.read_bytes())
        mutate(c)
        p = BUILD / f'ctx-{name}.bin'
        p.write_bytes(bytes(c))
        rc, so, se, data = run(binp, [p, 6, 6, 2, 0, 3])
        rows.append({'case': 'context-' + name, 'context_sha256': sha(bytes(c)), 'want_exit': want, 'exit': rc,
                     'stdout': so, 'stderr': se, 'verdict': 'PASS' if rc == want and 'NL4R1' not in so else 'FAIL'})
    return rows


def _magic(c):
    c[0] ^= 0xff


def _expiry_before_issue(c):
    c[72:80] = c[64:72]


BAD_CONTEXTS = [
    ('bad-magic', _magic, 11),            # v.om lpvalid: lpbe32 a 0 != 0x4c545030 -> 11
    ('expiry-equals-issue', _expiry_before_issue, 18),  # v.om lpvalid: expiry <= issued -> 18
]


MUTANTS = {
    # name: (old, new) applied to boundary.om; each must turn at least one case to FAIL.
    'wrong-chunk-expectation': (b'sh = mul get BCFG 4 2\n', b'sh = mul get BCFG 4 0\n'),
    'no-receiver-match': (b'if not nlmatch p e\nret 49\n', b'if false\nret 49\n'),
    'execute-before-all-verified': (
        b'for i ulong 16\nput props add mul chunk 16 i get p i\nend\nend\n',
        b'nlexec p\nend\n'),
    'no-envelope-validation': (b'rc = lpvalid frame part ctx 0 1024 now 0\n', b'rc = 0\n'),
}


def main():
    src = (TD / 'boundary.om').read_bytes()
    report = {'omc': str(OMC), 'omc_sha256': sha(OMC.read_bytes()), 'boundary_om_sha256': sha(src),
              'context_sha256': sha(CTX.read_bytes()), 'parts': [str(p) for p in PARTS],
              'prefix': 'neuralese/host-proposal.om lines 1..144', 'prefix_sha256': sha(prefix())}
    if report['omc_sha256'] != OMC_SHA:
        print('omc pin mismatch', file=sys.stderr)
        return 3
    c, b = compile_src('boundary', compose(src))
    report['compile'] = c
    if c['compile_exit'] != 0:
        print(json.dumps(report, indent=2))
        return 1
    report['cases'] = exercise(b)
    report['mutants'] = []
    for name, (old, new) in MUTANTS.items():
        if src.count(old) != 1:
            report['mutants'].append({'mutant': name, 'error': 'anchor not unique'})
            continue
        mc, mb = compile_src('mutant-' + name, compose(src.replace(old, new)))
        entry = {'mutant': name, 'compile': mc}
        if mc['compile_exit'] == 0:
            rows = exercise(mb)
            entry['failed_cases'] = [r['case'] for r in rows if r['verdict'] == 'FAIL']
            entry['detected'] = bool(entry['failed_cases'])
        else:
            entry['detected'] = False
        report['mutants'].append(entry)
    report['all_cases_pass'] = all(r['verdict'] == 'PASS' for r in report['cases'])
    report['all_mutants_detected'] = all(m.get('detected') for m in report['mutants'])
    out = TD / 'evidence/boundary-exercise.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    for r in report['cases']:
        print(r['verdict'], r['case'], r['exit'], r['want_exit'], repr(r['stdout'][:90]))
    for m in report['mutants']:
        print('MUTANT', m['mutant'], 'compile', m.get('compile', {}).get('compile_exit'), m.get('compile', {}).get('compile_stderr', '').strip(),
              'detected', m.get('detected'), m.get('failed_cases'))
    return 0 if report['all_cases_pass'] and report['all_mutants_detected'] else 2


if __name__ == '__main__':
    sys.exit(main())
