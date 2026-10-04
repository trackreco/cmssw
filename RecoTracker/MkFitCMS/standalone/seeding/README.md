# seeding/ -- the mkFit seeder's standalone driver

`seedsurf` runs the mkFit seeder (`MkSeeder`, `SeedChain`, `SeedChainFinder` in MkFitCore) on a
standalone sample, with truth, and keeps the double-precision reference finder the float one is
checked against.

| file | what |
|---|---|
| `seedsurf.cc` | the driver: options, the configuration of the two passes (one `SeedChain` per z side, the window tables, the fake cuts), the event loop through `MkSeeder`, truth, `--dump`, `--margins`, `--resid` |
| `SeedSurf.h` | the double-precision reference: `SurfChain` (the run on a `SeedChain`), the pattern finder, the ownership, `surf_eval()` |
| `SeedSurfBatch.h` | the old name of `SeedChainFinder`, and the `--chain-fast-check` comparison |
| `seedsurf-chain.sh` | the chosen configuration (2026-09-27): the chain, the window tables, cleaning N = 3; the reference of the identity checks |
| `seedsurf-v2p2.sh` | seeding for v2p2 at a named working point, `WP=` a directory of `wp/` (default 2026-10-04: `seedsurf-chain.sh` plus holes into OT1-P, d0_max 0.5 mm, zv 15 cm, fake score S < 1.5, S < 0.5 at \|eta\| >= 1.7) |
| `wp/` | the working points: the seeder's options and the finder settings each was measured with; `wp/README.md` |
| `windows-D121/` | the window tables for D121, as command-line options; `ot1p-hole.txt` holds B2 B3 B4 + OT1-P, B1 B2 B4 + OT1-P and B1 B3 B4 + OT1-P (q95 of true quads, events 30-99), for `--chain 1 --chain-holes-ot 1 --chain-inner-ot-only`, which `seedsurf-chain.sh` does not set |
| `truth2root.py` | `--truth` output into a ROOT file |

## Build and run

`seedsurf` is a target of the standalone build, so `./mymake` builds it next to `mkFit`. Run it
from the build directory:

```
./mymake -j 16
<this dir>/seedsurf-chain.sh --chain-batch --fk-score 0.75 --fk-ot2 1.5 --fk-shape \
    --num-events 5 --dump quads.txt --truth truth.txt
```

`B` (the build directory, default the current one), `SS` (the binary in it), `S` (the sample),
`GEOM` and `BIND` are read from the environment; `seedsurf-chain.sh` lists them.

## Acceptance of a change that should not move the quads

Run the same events and options with the binary before and after, `--dump` and `--truth` to two
files each, and `cmp` them. The import into MkFitCore was done that way (2026-10-01): byte-identical
for the batched chain with the nominal fake cuts on events 0-19 and for the double-precision chain
on events 0-1. An older build's binary can be run with the same script through `B` and `SS`.

## History

The seeder was developed standalone on branch `mkfit-seeding` of mkFit-external, 2026-09-21 to
09-30. Its research record, with what was tried and not kept and every measurement, is frozen there
in `mkfit-standalone-attic/mkfit-seeding/README.md`, beside the barrel-only `seedfind` line and the
geometry study `mkfit-standalone-seedgeom`.
