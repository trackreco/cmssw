# MkFinderV2p2 design notes

Design decisions behind `MkFinderV2p2` and the classes it uses (`MkBins`,
`LayerOfHits::HitInfo`). The code comments say what a piece of code does and
point to a section here for why. Each measurement quoted here names its sample,
its size and what was counted.

## Line pre-cut

**Code:** `MkFinderV2p2::select_hits()`, the per-candidate `pc_*` block and the
per-hit test before the prefetch. Constants `MkBins::PRECUT_*`, switches
`g_v2p2_precut_q` and `g_v2p2_precut_phi`, per-hit input
`LayerOfHits::hit_qbar_half_extent()`.

**What it does.** Every hit fetched from the binnor used to go straight into the
pre-selection batch, where it costs a Hermite solve onto its module plane and a
lookup of the module geometry. Only then is the real dq/dphi cut applied. The
pre-cut rejects most of those hits earlier, using only the cached `HitInfo`
array and four numbers per candidate.

Between its two layer crossings `m_sp1` and `m_sp2` the track is taken as a
straight line in (qbar, q) and in (qbar, phi). Here qbar is r in the barrel and
z in the endcap. The line is evaluated at the hit's own qbar, and the hit is
dropped when its q or phi is further from the line than a tolerance that is
looser than the real cut.

**Tolerances**, with g the slope of the line:

- q: `PRECUT_DQ_SLACK * dq_trk_fac * dq_track * (1 + g^2) + dq_hit_fac * hit_q_half_length + PRECUT_QBAR_FAC * |g| * hit_qbar_half_extent`
- phi: `PRECUT_DPHI_SLACK * dphi_trk_fac * dphi_track + hit term of the real cut + PRECUT_QBAR_FAC * |g_phi| * hit_qbar_half_extent`

The tolerances are written in terms of the real cut's own factors, so the
pre-cut stays looser when those factors change.

**Why each term.**

- `(1 + g^2)` equals 1/sin^2(theta) for a track from the origin. It references
  the track's q error to the layer surface. The real cut does the same per hit
  (`surface_referenced_dq()`). Without it the pre-cut would drop hits that the
  real cut accepts at high |eta|.
- `hit_qbar_half_extent` is `hl_fac * sigma_r` of the hit, in the barrel. The
  qbar of a hit is its centroid radius. For a strip on a tilted module the true
  crossing lies somewhere along the strip, and the strip direction has a radial
  component, so the centroid radius is uncertain by up to L sin(tilt). Through
  the slope of the line that becomes a shift in q and in phi. In the endcap
  qbar is z, and the same term is not needed, so it is skipped there on a
  per-layer branch.

**Offline estimate that justified it.** `val_qprecut_*` in
`standalone/RdfTrace/ValProp.cc` replays the pre-cut over the trace. It counts,
for each scanned candidate-hit pair, whether the pre-cut would reject it and
whether the real cut accepts it. A false reject is a pair the real cut accepts
and the pre-cut drops. Production window, trace build:

| | forward: pairs | rejected | false rejects | inward: pairs | rejected | false rejects |
|---|---|---|---|---|---|---|
| q, slack 2 (as built) | 11.9 M | 79.3 % | 82 | 1.83 M | 85.8 % | 0 |
| q slack 2, and phi slack 1.5 (as built) | 11.9 M | 89.8 % | 405 | 1.83 M | 89.1 % | 15 |
| q, slack 2, without the surface factor | 11.9 M | 80.7 % | 17610 | 1.83 M | 89.0 % | 10381 |
| q, slack 1 | 11.9 M | 82.2 % | 2278 | 1.83 M | 89.7 % | 554 |
| q, slack 1, without the qbar term | 11.9 M | 84.3 % | 87659 | 1.83 M | 90.1 % | 3364 |

Forward is 5 events of `ttbar-PU200-D121-C22-100ev.bin` with CMSSW seeds.
Inward is 10 events of `trackingNtuple_HLT_2026_March.bin` with pT5 seeds whose
pixel hits were removed. The real cut accepts 8.2 % and 7.6 % of the pairs. All
the q false rejects without the qbar term are in the tilted TBPS layers. The
phi false rejects sit mostly in TB2S.

**Physics, with the pre-cut in the finder.** Paired against the same
configuration with the pre-cut off, production window:

- forward, 30 events of `ttbar-PU200-D121-C22-100ev-rt.bin`: found tracks,
  fakes, duplicates and the momentum-resolution width identical to the unit,
  with q alone and with q and phi;
- inward, 50 events of chopped pT5: chopped hits recovered, fully recovered
  tracks and the truth-matched hit efficiencies identical to the unit, with q
  alone and with q and phi.

In one forward event the pre-cut rejects 2.87 M of 3.18 M scanned pairs
(90.4 %). The 306 k pairs left go through the plane solve, and 79 % of them
pass the real cut.
