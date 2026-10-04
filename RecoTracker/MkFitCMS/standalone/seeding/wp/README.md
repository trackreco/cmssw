# wp/ -- the working points of seeding v2p2

One directory per working point. Each holds the seeder's options and the finder settings the working point was
measured with. A working point is never edited once measured; a change makes a new directory.

| file | what | used by |
|---|---|---|
| `seedsurf.opts` | seedsurf options on top of `seedsurf-chain.sh`; a token `@path` is replaced by that file (relative to `seeding/`) | `seedsurf-v2p2.sh` (`WP=`), `wp/opts.sh` |
| `finder.json` | the `--json-patch` of mkFit's iteration 0 | the drivers in `cmssw_20_mkseed/v2p2-seeds/` (`FWP=`) |
| `finder.shell` | Shell commands, one per line, run before the first event; compiled Shell setters only, since cling does not resolve `Config::V2p2` | as above |

A file that is absent means the plugin's default: no patch, no command.

The default of `seedsurf-v2p2.sh` is 2026-10-04 (maintainer, 2026-10-04); it was 2026-10-02 before.

## 2026-10-02: agreed with the maintainer

- Seeder: holes into OT1-P, d0_max 0.5 mm, beam region +-15 cm in z, fake score S < 0.75, S < 0.5 where |eta| >= 1.7.
- Finder: the standalone plugin's defaults. The track scorer is `phase2:LstIntoPix` and there is no flagged-seed cut.
- Measured, default val_eff rule (2026-10-02): v2p2 + loose 82.74 %, 4675 fakes. Seeding 192.1 ms/ev and v2p2 on
  its quads 245.9 ms/ev on phi3 (`v2p2-seeds/time-phi3/v2p2/README.md`).

## 2026-10-04: loosened, with the flagged-seed cut (the default since 2026-10-04)

- Seeder: as 2026-10-02 with S < 1.5 where |eta| < 1.7.
- Finder: production's track scorer `phase1:default`; a final track whose seed has a fake score >= 0.35 is kept only
  with >= 4 hits added to the seed's (`StdSeq::remove_flagged_seed_tracks`).
- Measured with CMSSW's loose selection after the finder, MTV rule, 30 events: 90.92 %, 829 fakes; the seeds of
  2026-10-02 with this finder 90.40 %, 840 fakes; production's initialStep + highPtTripletStep 92.90 %, 3665 fakes
  (`v2p2-seeds/loosen/README.md`).
- Timed on phi3 (51 events, `v2p2-seeds/time-phi3/wp/README.md`): seeding 197.2 ms/ev, v2p2 on its quads 287.6 ms/ev;
  the seeds of 2026-10-02 191.4 and 270.2 ms/ev with this finder.

## 2026-10-04-gapmap: candidate, not the default

- Seeder: 2026-10-04 plus a start across B2 (B1 B3, `--chain-start-gap-barrel`) where the line crosses B2 in a gap
  between modules (`--chain-gap-map 0.02`, SensorGapMap, margin 0.2 mm). Finder: as 2026-10-04.
- Measured with CMSSW's loose selection, MTV rule, 30 events: 91.85 %, 832 fakes (barrel 93.89 %, production 94.08 %).
- Timed on phi3: seeding 233.4 against 203.4 ms/ev for 2026-10-04 in the same job, v2p2 290.1 ms/ev
  (`v2p2-seeds/time-phi3/wp/README.md`, third section; the study: `v2p2-seeds/adv/README.md`).
