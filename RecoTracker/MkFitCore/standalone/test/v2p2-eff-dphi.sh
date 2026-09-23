#!/bin/bash
# dphi pre-selection scan on the CURRENT forward configuration (in-layer, cap 3).
#
# The cut (MkFinderV2p2.cc:1330) is
#
#   ddphi < dphi_trk_fac * dphi_track + hit_dphi_fac * HIT_PHI_HALF_EXTENT
#
# and unlike the dq side it carries NO single global factor: the track term runs
# at 1x and the hit term is a flat DDPHI_PRESEL_FAC * 0.0123 = 0.0246 rad,
# detector-wide, with no per-hit, per-layer or geometry dependence. That flat
# term is ~190x the across-strip pitch term at TB2S radii, so the phi window is
# hugely over-generous where it should be the DISCRIMINATING cut -- the mirror
# image of dq, which is over-generous where it can only ever be containment.
#
# THREE FACTORS, NOT ONE, and the third is not bookkeeping. The cut can only
# reject hits the BINNOR already fetched, and the binnor opens
#
#   dphi_trk_fac * dphi_track + bin_dphi_fac * HIT_PHI_HALF_EXTENT
#
# with bin_dphi_fac = PHI_BIN_EXTRA_FAC = 2.75 by default. So raising
# hit_dphi_fac above bin_dphi_fac is a SILENT NO-OP, and the resulting flatness
# is an artefact of the fetch rather than a statement about physics. Every
# widening point below raises the binnor with it. (The track term needs no such
# care: dphi_trk_fac multiplies it in both places.)
#
# Paired: same events, same seeds, one val_dphi() call apart.
#
#   usage:  v2p2-eff-dphi.sh [n_events] [sample] [out_prefix]
set -e
cd /foo/matevz/mic-dev/current/src/standalone
unset DISPLAY
export LD_LIBRARY_PATH=.
N=${1:-30}
SAMPLE=${2:-/foo/matevz/mic-dev/ttbar-PU200-D121-C22-100ev-rt.bin}
OUT=${3:-eff-dphi}
T=../RecoTracker/MkFitCore/standalone/test

# label:trk_fac:hit_fac:bin_fac -- bin_fac must be >= hit_fac.
# "def" reproduces today's constants exactly and is the paired reference; it
# must also reproduce the dq3.0 point of v2p2-eff-dq.sh (18846 found over 30
# events), which is the check that the refactor is bit-neutral.
CFGS=${CFGS:-"def:1.0:2.0:2.75 hit1.0:1.0:1.0:2.75 hit0.5:1.0:0.5:2.75 hit4.0:1.0:4.0:4.50 trk2.0:2.0:2.0:2.75 trk0.5:0.5:2.0:2.75"}

CMD=(./mkFit --geom CMS-phase2 --seed-input cmssw --read-cmssw-tracks --input-file "$SAMPLE"
     --num-events "$N" --num-thr 1 --build-mimi --build-mimi-v2p2 --shell
     --shell-command 'gROOT->SetBatch(kTRUE)'
     --shell-command "gROOT->ProcessLine(\".L $T/val-prop.C\")"
     --shell-command 'val_eff_reset()'
     --shell-command 'val_eff_ref("def")')

for cfg in $CFGS; do
  IFS=: read -r lab trk hit bin <<< "$cfg"
  CMD+=(--shell-command "val_dphi($trk, $hit, $bin)")
  for ((i=1;i<=N;i++)); do
    CMD+=(--shell-command "s.GoToEvent($i)"
          --shell-command 's.ProcessEventStd()'
          --shell-command "val_eff_ev(s.event(), \"$lab\")")
    if [ "$lab" = "def" ]; then
      CMD+=(--shell-command 'val_eff_cmssw_ev(s.event(), "cmssw_V1")')
    fi
  done
done
CMD+=(--shell-command "val_eff_report(\"$OUT\")")
echo .q | "${CMD[@]}"
