# MkFinderV2p2 design notes

Design decisions behind `MkFinderV2p2` and the classes it uses (`MkBins`,
`LayerOfHits::HitInfo`, `V2p2Score.h`, `V2p2Config.h`). The code comments say
what a piece of code does and point to a section here for why. Each
measurement quoted here names its sample, its size and what was counted.

Contents:

1. [Samples](#samples)
2. [Configuration](#configuration)
3. [Layer processing](#layer-processing)
4. [Candidate pickup and stopping cuts](#candidate-pickup-and-stopping-cuts)
5. [Layer crossings and the Hermite cubic](#layer-crossings-and-the-hermite-cubic)
6. [Within-sensitive-region verdict](#within-sensitive-region-verdict)
7. [Search window](#search-window)
8. [Hit extents](#hit-extents)
9. [Line pre-cut](#line-pre-cut)
10. [Reduction and hit ordering](#reduction-and-hit-ordering)
11. [In-layer combinatorial search](#in-layer-combinatorial-search)
12. [End-of-layer selection](#end-of-layer-selection)
13. [Layer-step score](#layer-step-score)
14. [V2 compatibility](#v2-compatibility)

## Samples

The measurements below use four samples. They are named once here and by tag
afterwards.

- **F30**: forward search, 30 events of `ttbar-PU200-D121-C22-100ev-rt.bin`,
  CMSSW initialStep seeds, efficiency against the MTV-like denominator (29759
  selected sim tracks). Comparisons are paired: same events, same seeds, one
  process.
- **I50**: inward search, 50 events of `trackingNtuple_HLT_2026_March.bin`, pT5
  seeds with their pixel hits removed ("chopped pT5"). The search has to find the
  removed hits again. Counted: chopped hits recovered, out of 211435, matched by
  exact (layer, index); and tracks with all their chopped hits recovered, out of
  43358. Neither count uses truth matching.
- **T100**: timing, 100 events of `ttbar-PU200-D121-C22-100ev.bin`, single
  thread, build without ROOT and without tracing, minimum of three repetitions.
  The quoted time brackets `(builder.*FindTracks)(IT_FwdSearch)` only.
- **H50**: 50 events of `trackingNtuple_HLT_2026_March.bin`, used for detector
  properties such as hit multiplicities per layer crossing.

## Configuration

**Code:** `V2p2Config.h`, namespace `Config::V2p2`, groups `Window`, `PreCut`,
`InLayer`, `Policy`, `Score` and `Diag`.

Each knob is declared once, with its default, through `MKFIT_V2P2_KNOB`. The
CMSSW build gets a `constexpr`, so there is no mutable global state and branches
on the switches fold away. The standalone build gets a mutable variable,
defined in `V2p2Config.cc`, so that `mkFit` options and the shell drivers can
scan a knob without a rebuild.

The knobs are meant to move into `IterationParams`, per iteration and settable
from the JSON configs. The defaults in `V2p2Config.h` then become the
`IterationParams` defaults.

The policy counters (`V2p2PolicyCounters`, printed with `--quality-val`) exist
in the standalone build only. The finder increments them through
`V2P2_COUNT()`, which expands to nothing in CMSSW.

## Layer processing

**Code:** `MkFinderV2p2::process_layer()`, `process_layer_batch()`, the
`LayerBatch` and `HitBatch` structs in `MkFinderV2p2.h`.

`process_layer()` pulls CombCandidates into the layer, one `PrimTCandRep` per
live TrackCand, and drains the resulting queue NN candidates at a time. Pulling
in stops once the queue holds NN entries, so the Matriplex batches run full.

`process_layer_batch()` takes one batch through the layer in phases:

1. `prop_to_layer_edges()`: propagate each candidate to the two bounding
   surfaces of the layer.
2. `determine_search_windows()`: build the window covariance, the dphi and dq
   windows, the WSR verdict, the binnor ranges and the Hermite cubic.
3. `select_hits()`: walk the bin ranges, apply the line pre-cut, and send the
   survivors to `preselect_hit_batch()`, which solves the cubic onto each hit's
   module plane and applies the dq/dphi cut.
4. `prepare_kalman_workload()`: drain the per-candidate priority queues into one
   step-ordered list.
5. Either `expand_in_layer()` (the in-layer combinatorial search) or
   `kalman_update()` plus `process_kalman_results()` (one best hit per layer).

Two batch widths are in play. `LayerBatch` is NN candidates wide and indexed by
`i`. `HitBatch` is NN (candidate, hit) pairs wide and indexed by `h`, with
`prim_idcs[h]` naming the candidate.

Each phase keeps its own Matriplex batches full: pre-selection batches pairs
across candidates, and the Kalman stage batches lanes across candidates and
across expansion nodes. What this layout does not do is pipeline across phases,
or retire a CombCandidate before the end of the layer. The selection runs in
`end_layer()` because one CombCandidate's candidates can straddle a batch
boundary. Retiring a CombCandidate as soon as its last batch completes is
possible and not built.

## Candidate pickup and stopping cuts

**Code:** `awaken_candidates()`, `begin_next_Ccrep_in_layer()`,
`stop_cuts_at_pickup()`, `fake_hit_index()`, `end_layer()`.

**Pickup.** A dormant CombCandidate wakes up when the plan reaches its pickup
layer, matched against either sub-layer of a paired entry. A seed whose last hit
sits in the layer being searched needs no special case. Its state is at that
hit, so the hit sits at dalpha = 0 and everything further along at dalpha > 0,
and the forward-only expansion never returns to it. `propagate_to_r()` picks the
crossing with the smaller |alpha|, so the entry crossing of such a candidate is
behind it, at negative dalpha, and the window also covers the part of the shell
behind the candidate. That costs scanned hits and gives no wrong answers.

**Stopping cuts.** Both cuts read only the candidate's own state, so they are
applied at pull-in, before any work is spent on the layer:

- `pT < minPtCut` from `IterationParams`;
- the looper stop, forward search only: pT below `Policy::looper_max_pt`, r above
  `Policy::looper_min_r`, and the transverse angle between position and momentum
  past `Policy::looper_max_angle` (pi/2 - 0.2, 78.5 deg). Past that angle the
  track crosses modules at grazing incidence, clusters get wide, and further
  hits degrade the measurement.

The angle test is on |posPhi - momPhi| with both angles wrapped to (-pi, pi]. An
angle A past the limit therefore appears as dphi > A, or as dphi < 2 pi - A when
the pair straddles the branch cut. The upper bound is derived from the lower one
in code. V1 writes it as the literal 4.512, which is pi + A and lets the
78.5-101.5 deg band through whenever the pair straddles. The test as written is
equivalent to cos(momPhi - posPhi) < sin(0.2).

A stopped candidate records why, in `TrackCand::StopReason_e`, because loopers
are of interest downstream (the phase-2 timing layer and HGCal).

**Holes.** A candidate that takes no hit in a layer gets a fake HoT chosen by
`fake_hit_index()`, in V1's order: the hole limits (`maxHolesPerCand`,
`maxConsecHoles`) decide between a miss and a stop, and then the WSR overrides
with an edge or gap HoT, which do not count against the limits.

**Retiring.** `end_layer()` marks a CombCandidate finished once every TrackCand
under it has stopped.

## Layer crossings and the Hermite cubic

**Code:** `MkBins::prop_to_limits_in_order()`, `prop_to_layer_edges()`,
`preselect_hit_batch()`, `mini_propagators::Hermite3D` and `Hermite3DOnPlane`.

The candidate is propagated to the layer's two bounding surfaces in the order
the track meets them: `m_sp1` is the entry crossing and `m_sp2` the exit. These
are parameter-only propagations (`mini_propagators`), closed form in both r and
z. `m_isp` is left at `m_sp1`.

A cubic Hermite curve through the two crossings, with their momenta as
tangents, is the trajectory model inside the layer. For each scanned hit,
`Hermite3DOnPlane` solves the cubic onto the hit's own module plane with one
Newton step, and the dq/dphi cut and the state handed to the Kalman update are
both taken from that point (`h3_state`). The full helix-plane intersection is
transcendental for a tilted module, and the cubic is what makes a per-hit plane
solve affordable. The residual distance to the plane after the step is recorded
in the trace as `d_plane_h3`.

`MKFIT_TRACE_PROP_COMPARE` keeps a straight-line step onto the same plane as a
cross-check.

## Within-sensitive-region verdict

**Code:** `MkFinderV2p2::determine_wsr()`, `PrimTCandRep::m_wsr`.

The verdict is per candidate and per layer, and asks whether the segment the
track cuts through the layer's bounding shell lies in sensitive material. V1
tests one propagated point; v2p2 has both crossings and tests the segment.

- The two mini-propagator fail flags answer the radial half. Both failing means
  the track turns around before the layer: outside. Only the exit failing means
  it enters and turns around inside: at best an edge. Only the entry failing
  cannot happen, because the entry surface is the near one. In the endcap
  `propagate_to_z()` cannot fail, and the q test below carries the whole
  decision.
- `m_q_min` and `m_q_max` are the q extent of the segment. They are compared with
  the layer's q limits, widened by `Policy::wsr_n_sigma` sigma of the track's q
  error. Inside requires the whole widened segment to be inside. Erring towards
  edge is the safe side: an edge that was really inside loses one hole count,
  while an inside that was really an edge charges a hole against a candidate
  that crossed nothing.
- An endcap disc with an r hole gives a gap verdict when the segment touches the
  hole. No phase-2 layer sets `set_r_hole_range()` today, so this branch does not
  fire.

A candidate outside the layer is skipped: no bins are walked and no HoT is
added. The layer plans are deliberately inclusive (a transition plan is the
union over tracks), and without this every one of those layers would record a
hole. Measured on 20 events of the forward search on
`ttbar-PU200-D121-C22-100ev.bin`: the verdict declines 64.5 % of OT-barrel
layer searches and 23.8 % of forward ones, and the sim track has a hit in the
declined layer in 0.05 % and 0.30 % of those.

## Search window

**Code:** `MkBins::determine_bin_windows()`, `MkBins::find_bin_ranges()`,
`MkFinderV2p2::preselect_hit_batch()`, `surface_referenced_dq()`,
`Config::V2p2::Window`.

The pre-selection window has two parts: a binnor **fetch**, a range of (phi, q)
bins, and a per-hit **cut**. The invariant is that the fetch covers everything
the cut accepts. A hit the cut would accept but the fetch never pulled is lost
silently.

### The cut

```
ddq   < dq_trk_fac   * dq_trk     + dq_hit_fac   * hit_q_half_length
ddphi < dphi_trk_fac * dphi_track + dphi_hit_fac * hit_phi_half_extent
```

`dq_track` and `dphi_track` are 3 sigma of the track's position error, from the
position block of the covariance at `m_sp2` (see "Track covariance at the
layer" below). `dq_trk` is `dq_track` referenced to the hit's module surface.
Each residual is taken at the hit's own module plane.

### Track covariance at the layer

`MkBins::transport_position_cov()` transports the position block of the
covariance from the previous hit to `m_sp2`, at the fixed path length the
mini-propagator reached it with. Only the position block is needed. At fixed
path length s the position is x = x0 + f(ipt, phi, theta; s), so

```
C_pos(s) = J C0 J^T,   J = [ P_in | dx/d(ipt, phi, theta) ]   (3 x 6)
```

with the derivatives in closed form from quantities the mini-propagator already
holds. With the helix in the turning angle a, k = 1/inv_k and p in GeV,

```
x = x0 + k (px sin a - py (1 - cos a)),  y = y0 + k (py sin a + px (1 - cos a)),
z = z0 + s cos(theta),                   a = s sin(theta) ipt inv_k
```

so, with (dx, dy) the displacement and p_end the transverse momentum at `m_sp2`,

```
d(x,y)/dphi   = (-dy, dx)
d(x,y)/dipt   = (-(dx, dy) + a k p_end) / ipt = k (px f1 - py f2, py f1 + px f2) / ipt
d(x,y)/dtheta = a k p_end cot(theta),     dz/dtheta = -a k / ipt
f1 = a cos a - sin a,   f2 = a sin a - (1 - cos a)
```

with p at the previous hit in the second form of the ipt column. As in
`errPropFromPathL_impl()` the result is curvilinear at both ends: `P_in`
projects the starting position onto the plane normal to the momentum there,
and the result is projected onto the plane normal to the momentum at `m_sp2`.
The dq surface reference below depends on that. The first form of the ipt
column is a difference of two O(s) terms that nearly cancel for a stiff track:
it lost 25 % in sigma_y at pT 137 GeV. The second form is used, with f1 and f2
from their series below |a| = 0.25, where they are O(a^3) and O(a^2).

No material enters. Material added during a step changes only the angular
terms, so none of it reaches the position block within that step.

This replaced a full `propagateHelixToPlaneMPlex` call, of which only the same
six elements were read. Against that call with uniform B, over 200508 lanes,
the sigmas agree to 7e-5 and the correlations to 1.6e-4 at the worst lane.
Against it with the parametric field at the starting point they differ by
0.4 % at p99. The transport uses the same uniform B as `m_sp1`, `m_sp2` and the
Hermite cubic. On 20 events of `ttbar-PU200-D121-C22-100ev.bin` the quality-val
track counts were identical in the default and two varied configurations. The
block costs 0.23 s against 0.68 s over those 20 events in a trace build, about
5 % of the build time.

**Two factors per coordinate.** Which term binds is a property of the layer.
The median `hit_q_half_length` (H50) is 0.0075 cm in the pixel barrel, 0.042 cm
on the P sensors and 0.80 cm on the S sensors of TBPS, and 2.51 cm in TB2S,
while the track term does not vary like that. Strips are containment-dominated
and pixels are covariance-dominated. One factor over both terms could not sit
in the right place for both, and a scan of it could not say which term it was
moving.

- `dq_hit_fac` multiplies the hit's own half-extent, so its floor is exactly
  1.0: below it the window no longer reaches the whole strip. The default 1.2
  keeps 20 % margin. Against 1.8, F30 gives -3 found tracks and -15 fakes, and
  I50 gives +55 recovered hits and -2 fully recovered tracks.
- `dq_trk_fac` is 1.5, in units of `dq_track`.

**dq surface reference.** The covariance is transported to a fixed path length,
not to a plane. The resulting q error describes where
the track is after travelling a distance s, not where it crosses the module. The
two differ by the ds degree of freedom. Sliding each point along the momentum
until it meets the surface is a linear map on the position block,

```
dx_s = (I - p^ n^T / (n^.p^)) dx
```

so the q variance becomes v^T C v with v = e_q - ((e_q.p^)/(n^.p^)) n^.
`surface_referenced_dq()` applies it with n^ the hit's own module normal. For a
radial track and a cylinder normal this amplifies sigma_q by 1/sin^2(theta).
With the module normal, TBPS modules, which face the interaction point, get
almost no correction, and tilted and flat layers use the same formula with no
branch. Measured on 20 events of chopped pT5, pixel barrel, as the ratio of the
measured residual core to the quoted sigma_q:

| \|eta\| | 0-0.8 | 0.8-1.6 | 1.6-2.0 | > 2.0 |
|---|---|---|---|---|
| without the reference | 1.64 | 4.66 | 17.0 | 46.0 |
| with the reference | 1.38 | 1.43 | 1.71 | 2.14 |

`MkBins::surface_reference_dq()` is the same correction with the layer's
cylinder or disc normal. It is off by default (`Diag::mkbins_surface_q`) and is
kept because it is the way a corrected `dq_track` reaches the trace.

**dphi jacobian.** `sigma_phi` comes from the xy covariance through
J = (-y, x)/r^2, so |J| = 1/r. One window covers the whole layer, so J is taken
at whichever crossing has the smaller radius, where `sigma_phi` is largest. The
ratio rout/rin is 1.33 at pixel layer 0.

**Per-hit phi extent and `dphi_trk_fac` = 2.** The hit term is the hit's own phi
extent from its covariance ([Hit extents](#hit-extents)), about 4e-5 rad in
TB2S. The cut is therefore carried by the track term, and the track term needs
a factor 2:

| `dphi_trk_fac` | F30 found | F30 fakes | I50 fully recovered tracks |
|---|---|---|---|
| 1.0 | -41 (4.2 sigma) | +87 | -582 |
| 1.5 | -2 | +10 | -29 |
| 2.0 | +1 | +2 | identical |

The reference row is the flat tolerance of 0.0246 rad that the per-hit extent
replaced. It is still available with `Window::phi_per_hit` off, as
`Window::dphi_flat_rad`.

Above |eta| 0.8 the factor also compensates a phi covariance that is too small.
The ratio of the measured phi residual core to the quoted `sigma_phi`, 10
events of the forward search on `ttbar-PU200-D121-C22-100ev.bin`, is 1.25 at
|eta| < 0.8, 2.20 at 0.8-1.6, 3.40 at 1.6-2.0 and 2.66 above. The cause is the
material description in the transition and forward regions. When that is
fixed, `dphi_trk_fac` should come down.

### The fetch

`find_bin_ranges()` derives the fetch from the cut, so a cut wider than the
fetch cannot be expressed. Two details:

- The cut is per hit, and the fetch runs before any hit is known, so the fetch
  uses the layer's largest hit extent (`max_hit_q_half_length()`,
  `max_hit_phi_half_extent()`).
- The phi range comes from the axis helper (`LayerOfHits::phiRangeBins()`),
  which returns a half-open range that covers the upper edge, with the mask
  applied after the increment. A hand-rolled pair of `phiBinChecked()` calls
  misses the bin holding the upper edge on every range. The phi half-width is
  clamped below pi first: a range on a circle is an arc, and a half-width of pi
  or more wraps to a small arc instead of the full circle.

Beyond the cut, the fetch adds whole bins on the bin index, so the margin
carries no float-to-bin rounding:

- `Window::phi_extra_bins` = 0. Against 1, F30 gives -1 found track and I50
  gives +1 recovered hit. One spare bin cost 20 % of build time before the line
  pre-cut (T100: 60.79 s against 48.38 s), since every fetched hit costs a
  plane solve per candidate.
- `Window::q_extra_bins` = 1. The per-hit q cut is surface-referenced and the
  fetch uses the raw `dq_track`, so at high |eta| the cut reaches past the
  fetch. With 0, I50 loses 1373 of 43358 fully recovered tracks, all in
  pixel-barrel hits of tracks that also cross the discs. F30 is unaffected. A
  fetch widened by the surface reference would remove the need for the spare
  bin. With the line pre-cut in, a spare bin costs little, since most fetched
  hits are rejected before the plane solve.

## Hit extents

**Code:** `LayerOfHits::HitInfo`, `hit_phi_half_extent_of()` and
`hit_r_half_extent_of()` in `HitStructures.cc`.

`HitInfo` caches, per hit, the extents the window and the pre-cut use. All are
`hl_fac` times a sigma from the hit's own covariance, with `hl_fac` = 3 for
pixels (a Gaussian 3 sigma) and sqrt(3) for strips, where a crossing uniform
along a segment of half-length L has sigma = L/sqrt(3).

- `q_half_length`: `hl_fac * sqrt(ezz)` in the barrel. For a strip along unit
  vector u the covariance is (L^2/3) u u^T, so this is L |u_z|, the strip's
  projection onto z. Module tilt is therefore already included. In TB2S, where
  modules are flat, the H50 median is 2.5125 cm against a nominal 2.5. In the endcap
  it is `hl_fac * sqrt(exx + eyy)`, which is sigma_r^2 + r^2 sigma_phi^2. That is
  right for radial strips and a factor sqrt(2) too large for endcap pixels,
  where sigma_x and sigma_y are similar.
- `phi_half_extent`: from sigma_phi^2 = (y^2 exx - 2xy exy + x^2 eyy) / r^4, the
  across-strip error seen from the origin. It needs no barrel/endcap branch. For
  a TB2S strip it is expected to be `hl_fac * pitch / sqrt(12)` / r, i.e.
  45 um / r for a 90 um pitch.
- `qbar_half_extent`: `hl_fac * sigma_r` in the barrel, 0 in the endcap. It is
  used by the line pre-cut for tilted strips.

## Line pre-cut

**Code:** `MkFinderV2p2::select_hits()`, the per-candidate `pc_*` block and the
per-hit test before the prefetch. Switches and slack factors in
`Config::V2p2::PreCut`, per-hit input `LayerOfHits::hit_qbar_half_extent()`.

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

- q: `dq_slack * dq_trk_fac * dq_track * (1 + g^2) + dq_hit_fac * hit_q_half_length + qbar_fac * |g| * hit_qbar_half_extent`
- phi: `dphi_slack * dphi_trk_fac * dphi_track + hit term of the real cut + qbar_fac * |g_phi| * hit_qbar_half_extent`

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
configuration with the pre-cut off, production window: F30 found tracks, fakes,
duplicates and the momentum-resolution width are identical to the unit, and so
are the I50 counts and truth-matched hit efficiencies, with q alone and with q
and phi.

**Time.** T100: 47.95 s without the pre-cut, 35.36 s with q, 33.81 s with q and
phi. V2 (`selectHitIndicesV2`) takes 62.95 s on the same events.

In one forward event the pre-cut rejects 2.87 M of 3.18 M scanned pairs
(90.4 %). The 306 k pairs left go through the plane solve, and 79 % of them
pass the real cut.

## Reduction and hit ordering

**Code:** `PrimTCandRep::m_pqueue`, `preselect_hit_batch()`,
`prepare_kalman_workload()`.

**Reduction.** Hits that pass the cut go into a bounded priority queue keyed on
ddphi, one queue per sub-layer, capped at `InLayer::max_presel_hits`. One queue
per sub-layer gives each sensor of a pair its own budget, so a busy sensor
cannot crowd out the other. The cap sits upstream of the in-layer search and
bounds what it can see. Overlap availability varies strongly across the
detector, so the cap should become per layer.

**Ordering.** The queues drain into one list per candidate, `m_layer_hits`,
sorted by `dir * dalpha`. dalpha is monotone in path length for all hits of one
candidate, since they are reached from the same state with the same curvature,
and the sign factor turns it into path order for either search direction. ddphi
decides which hits survive; path order decides the order the search walks them.
The sort is done once per candidate. A Kalman update moves the trajectory by
about the hit resolution, so it can only swap hits that are already that close.

The sub-layers of a TBPS pair are radially interleaved (the tilted modules make
the partner sensor sit at larger or smaller r module by module), so path order
cannot be assumed from the layer index. Per-hit precision is not an ordering
key: `q_half_length` already distinguishes a P macro-pixel from an S strip,
and it enters the score.

The trace records `sub_rank` (rank by ddphi within the sub-layer) and
`full_rank` (across both), for measuring pre-selection against truth.

## In-layer combinatorial search

**Code:** `expand_in_layer()`, `harvest_sec_nodes()`, `SecTCandRep`,
`MkFinderV2p2::m_sec_arena`, `InLayer::comb`.

**What it replaces.** The best-hit path runs one Kalman update per pre-selected
hit, all from the candidate's incoming state, and keeps the lowest chi2. That
takes at most one hit per layer. On H50 a track crossing a TBPS, TB2S or TEDD
layer pair leaves both sensors' hits in 91-94 % of crossings, and 26-28 % of
crossings carry three or more hits, because an overlapping module is itself a
stack.

**What it does.** A path through the layer is an increasing sequence of
positions in `m_layer_hits`. The expansion grows all paths of depth d before any
of depth d+1. Depth 0 starts from the candidate's state, and the Hermite has
already solved each crossing, so the propagation is handed the path length.
Deeper nodes start from an updated state, for which no crossing is known, and
propagate-to-plane solves it. The one-point Hermite
(`Hermite3D::calculate_coeffs(sp, inv_k, dalpha)`) would be cheaper there and
is not yet used.

Forward-only traversal of an ordered list reaches every subset of the hits
exactly once. Taking both hits of an overlap therefore needs no special case,
and there is nothing to de-duplicate. Path order also puts the Kalman updates
in the order the track meets the hits.

A second hit from the same module as an earlier hit of the path is not taken.
That is a split cluster or another track's hit, and a split cluster is one
measurement: taking it twice shrinks the covariance without adding information.

`InLayer::max_sec_depth` (4) caps the hits per path. Forward, 10 events of
`ttbar-PU200-D121-C22-100ev.bin`, found tracks stop changing at depth 3. Inward
on 20 events of chopped pT5, fully recovered tracks go from 82.4 % at depth 2
to 90.8 % at depth 4.

**Storage.** Nodes live in one `std::vector<SecTCandRep>` per finder, reached by
index, with a single parent index each. A node exists to be walked back from,
so a surviving leaf can register its hits into the CombCandidate; nothing needs
to walk forward. `end_layer()` clears the vector and keeps its capacity. There
is no free list: a node with live children must not be reused.

The arena is per finder and not per CombCandidate because one Kalman batch
draws lanes from several candidates and CombCandidates. Breadth-first by depth
keeps every parent at a lower index than its children, so a forward sweep of
the arena is a valid topological order.

A `SecTCandRep` carries a full `TrackState` and lives for one layer. A
`HoTNode` in `CombCandidate::m_hots` is 12 bytes and lives for the event. That
difference is why the two are separate.

**Effect.** F30, cap 3, against the best-hit path: found tracks 18455 to 18846,
fakes 3217 to 2765. Truth-matched hits per found track go from 10.78 / 11.55 /
11.32 to 12.70 / 14.32 / 14.29 in barrel / transition / endcap. Inward on 20
events of chopped pT5, at depth 2: chopped hits recovered 75.7 % to 92.2 %,
fully recovered tracks 52.8 % to 82.4 %.

`maxCandsPerSeed` 3 is the production value with the search on. From 3 to 6
F30 gains 39 found tracks. Before the line pre-cut, T100 build time with the
search on was 60.79 s at cap 3 and 76.45 s at cap 4.

## End-of-layer selection

**Code:** `select_and_materialise()`, `offer_best_short()`, `SelEntry`.

Everything that could continue a CombCandidate competes in one list on one
score:

- every in-layer path of every `PrimTCandRep`;
- every `PrimTCandRep` as the hole it would record, including those that did
  find paths;
- every TrackCand that did not become a `PrimTCandRep` (stopped, skipped at
  pull-in), at its unchanged score.

The list is partially sorted and the top `maxCandsPerSeed` survive. One sort is
possible because the score is additive over layer steps
([Layer-step score](#layer-step-score)): paths from different candidates are on
one scale. It runs at the end of the layer because one CombCandidate's
candidates can be spread over several batches.

**The hole competes with the hits.** A candidate may prefer a well-fitting wrong
hit to a hole, and no score of a single hit can tell. Offering the hole as a
competitor defers the decision until the paths can be compared. A candidate
outside the layer (WSR) is not a hole and competes unchanged.

**Reserve a hole slot** (`InLayer::reserve_hole_slot`, off by default). Taking a
hit always shrinks the covariance. When every survivor took a hit, no branch
remains that allows an earlier hit to have been wrong. This option keeps the
best decliner in the last slot. It is a beam policy, not a score term.

**Best short** (`InLayer::best_short`, off by default, outward only). A stopped
candidate cannot be extended, so it leaves the beam and the best one is kept on
the CombCandidate, the way V1's `CandCloner` does; `mergeCandsAndBestShortOne`
re-inserts it at the end if it still wins. Inward the trailing end of the track
is its head, and a truncated candidate is not a shorter track.

Survivors are built as copies before the CombCandidate is touched, because
several survivors can descend from one TrackCand.

`TrackCand::score_` holds the layer-step score during finding. At the end of the
search it is overwritten by `track_score_func`, which is what output and
cross-seed duplicate removal use. The two scores are never compared with each
other.

## Layer-step score

**Code:** `V2p2Score.h`, `LayerStepFeatures`, `Config::V2p2::Score`.

This score is separate from `track_score_func`, which stays as it is for V1/V2
and for the final output. It differs in three ways.

1. **Additive over layer steps.** A candidate's score is the sum of one term per
   layer. That is what makes the end-of-layer selection a single flat sort.
   `track_score_func` is additive only for some parameter choices: with the
   default scorer the hit bonus is linear in the hit count, so the total is
   quadratic, and the two hole penalties differ, so reclassifying a hole from
   tail to inside changes the total.
2. **No tail holes.** `TrackCand` reclassifies holes as inside or tail when a
   later hit is found, because the global formula is evaluated once at the end.
   Scoring each step when it happens has no such question. What the tail
   penalty approximated depends on direction: outward, trailing holes are at
   large radius where a track may leave the detector; inward, they are at small
   radius, where the track must have come from. The score has separate forward
   and backward parameter sets (`Score::fwd`, `Score::bkw`) for this; they carry
   the same numbers today.
3. **A feature struct.** `LayerStepFeatures` carries what a layer step did: hits
   taken, chi2, the best hit's `q_half_length`, ln det V of the residual
   covariance, local hit density, hole kind, the layers stepped between, and the
   candidate's state. New features do not change a function signature, and the
   struct can be written to the trace for fitting a score offline.

`layer_from` is the layer of the candidate's last found hit, so the step names
the propagation that happened. A candidate that missed a layer steps across two.

**Two forms.** `Score::mode` 0 is linear: a bonus per hit and per overlap hit,
minus a chi2 weight, minus a penalty per hole kind. Mode 1 is a log-likelihood
ratio against "this layer produced no hit":

```
take hit j :  ln(eps/(1-eps)) - ln(2 pi) - chi2_j/2 - ln(det V_j)/2 - ln rho
take none  :  0
```

with eps the per-layer hit efficiency (`hit_eff`), V the 2x2 residual
covariance and rho the local hit density. The hit-versus-hole break-even is then
derived rather than tuned. It assumes chi2 is trustworthy, which it is to the
extent the covariance is.

**Term ablation** (`Score::use_rho`, `use_detv`). rho and eps enter the
likelihood the same way, so replacing ln rho by a constant equal to its measured
mean removes only its variation. On F30 in mode 1 with eps 0.99, removing the
variation of rho costs 114 found tracks and removing that of det V costs 67.
Production uses mode 0.

## V2 compatibility

**Code:** `MkFinder::selectHitIndicesV2()`.

V2 is the production hit selection in CMSSW. It keeps its own window code, the
local `Bins` struct, as in upstream CMSSW, and does not use `MkBins`, so the
v2p2 window work does not move it. Its text differs from upstream in two
places only, both following interface changes on this branch: the
`propagate_to_plane()` call takes the module position and normal instead of a
`ModuleInfo`, and three renamed helpers inside `RNT_DUMP_MkF_SelHitIdcs`.

V2 does run on this branch's mini-propagators. `propagate_to_r()` is closed
form, where upstream iterates `Config::Niter` times without a convergence test,
and it clamps the target radius when the layer cannot be reached. The two agree
where the iteration converges.

The upstream window keeps two properties that the v2p2 fetch does not: the upper
phi edge of the bin range has no "+1", so the bin holding it is not scanned,
and a margin of 2.75 half-bins covers it; and the dphi jacobian is evaluated at
the track's incoming state rather than at the layer.
