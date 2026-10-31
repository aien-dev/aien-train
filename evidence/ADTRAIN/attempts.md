# Compiler refusals during development

These came from the pinned `seed/out/omc` (SHA-256 `62775d2a…3712b`). The
intermediate units were overwritten during iteration and are not retained;
only the refusal lines below were observed and recorded. The compiler was
never rebuilt or swapped.

| unit | refusal | cause | change |
|---|---|---|---|
| metadata + tape + `main` stub | `line 382: error 12` | register pool exhausted (16) in `tpmmf` | one matmul output moved into a per-element word `tpmmo` |
| same | `line 380: error 12` | `tpmmo` still held too many named values | inputs recomputed, products split into `x`, `y` |
| same | `line 470: error 12` | `tpbcast` copy loop | loops moved into `tprluf`, `tpbcf`, `tpsc`/`tpsr`; elementwise backward split into `tpbewa`/`tpbewb` |
| same | `line 627: error 12` | `tpgbo` nested expression | products split into `x`, `y` |
| metadata + tape + mlp + main | `line 854: error 7` | `int` compared with `long` in `mlid` | compare as `int` |

Meanings are from the omega MCP `explain_error` (omega-mv@e5b32567
`docs/one_layer/OMEGESE.md:492` for error 12; `:440-441` for error 7).
