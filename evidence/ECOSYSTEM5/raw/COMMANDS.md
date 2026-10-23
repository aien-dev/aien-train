# Executed host commands

Host-only developer commands, recorded2026-10-23. Generated ELF files stayed in
the handoff build directory. No executable binaries or shell scripts delivered.
Exit codes and stdout/stderr files are adjacent. These commands are not native
dataset or model execution claims.

```sh
git -C /home/drakestapleton/workspace/omega-learning652-lead-L775 show 84d03c5523f39ca1045d1c313c5f62597779462b:src/learning/benchmark/core.om > /home/drakestapleton/handoffs/ecosystem303/data5/build/core.om
git -C /home/drakestapleton/workspace/omega-learning652-corpus-L776 show d8d14c47647e49774f64b58e3c138335656f175d:src/learning/curriculum/admit.om > /home/drakestapleton/handoffs/ecosystem303/data5/build/admit.om
cat /home/drakestapleton/handoffs/ecosystem303/data5/build/core.om /home/drakestapleton/handoffs/ecosystem303/data5/build/admit.om /home/drakestapleton/workspace/aien-train-ecosystem5/native/ecosystem5/pack.om /home/drakestapleton/workspace/aien-train-ecosystem5/tests/ecosystem5/pack-test.om > /home/drakestapleton/handoffs/ecosystem303/data5/build/combined.om
/home/drakestapleton/workspace/omega-ecosystem660-bridge/build/bootstrap/omc < /home/drakestapleton/handoffs/ecosystem303/data5/build/combined.om > /home/drakestapleton/handoffs/ecosystem303/data5/build/pack-test 2> /home/drakestapleton/workspace/aien-train-ecosystem5/evidence/ECOSYSTEM5/raw/final-compile.log
chmod +x /home/drakestapleton/handoffs/ecosystem303/data5/build/pack-test
/home/drakestapleton/handoffs/ecosystem303/data5/build/pack-test > /home/drakestapleton/workspace/aien-train-ecosystem5/evidence/ECOSYSTEM5/raw/final-run.log
/home/drakestapleton/workspace/omega-ecosystem660-bridge/build/bootstrap/omc < /home/drakestapleton/workspace/aien-train-ecosystem5/tests/ecosystem5/mutate.om > /home/drakestapleton/handoffs/ecosystem303/data5/build/mutate 2> /home/drakestapleton/workspace/aien-train-ecosystem5/evidence/ECOSYSTEM5/raw/mutator-compile.log
chmod +x /home/drakestapleton/handoffs/ecosystem303/data5/build/mutate
```

Mutants used a direct shell loop for taskmode1,2,3. Each input was the ASCII
mode byte emitted by printf followed immediately by exact pack.om emitted by
cat. This input was piped to the native mutate executable. Each emitted
`raw/mutantN.om` replaced pack.om in the same four-file composition. The pinned
compiler produced build/mutantN; chmod and execution occurred only after its
compile exit0. Results are mutantN-generation.rc, mutantN-compile.rc and
mutantN-run.rc, with distinct compile/run logs. All three tuples are0/0/1.

The first rejected invocation was exactly:

```sh
/home/drakestapleton/workspace/omega-ecosystem660-bridge/build/bootstrap/omc /home/drakestapleton/handoffs/ecosystem303/data5/build/combined.om /home/drakestapleton/handoffs/ecosystem303/data5/build/pack-test
```

It returned16 because the compiler consumes stdin, not filename arguments.
The following compile attempts2..8 used the corrected stdin/stdout form with
the source snapshots named in the receipt. Compile2 used initial-combined.om;
compile3..7 each have their exact combined snapshot. Compile8 used the final
product module and the earlier274903-check test source; its compiler stderr
and exit are retained. Final source088258d includes the expanded test suite.

```sh
uname -srmo
file /home/drakestapleton/handoffs/ecosystem303/data5/build/pack-test
readelf -l -d /home/drakestapleton/handoffs/ecosystem303/data5/build/pack-test
/usr/bin/time -f 'elapsed_seconds=%e maxrss_kib=%M exit=%x' -o /home/drakestapleton/workspace/aien-train-ecosystem5/evidence/ECOSYSTEM5/raw/cost.txt /home/drakestapleton/handoffs/ecosystem303/data5/build/pack-test > /home/drakestapleton/workspace/aien-train-ecosystem5/evidence/ECOSYSTEM5/raw/cost-run.log
```
