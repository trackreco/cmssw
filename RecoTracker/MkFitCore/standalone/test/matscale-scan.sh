#!/bin/bash
# Is the residual isotropic covariance deficit (~1.4x in sigma, ~2x in variance)
# under-counted MULTIPLE SCATTERING?
#
# applyMaterialEffects samples ONE 1cm x 1cm (z,r) bin at the destination and
# applies it as a single thin scatterer with NO path-length scaling, so a long
# step under-counts. g_mat_scale multiplies radL (scattering only -- the
# energy-loss terms use hitsXi and are untouched). If the measured/quoted ratio
# walks to 1 near scale ~2, that is the answer, and the real fix is path-length
# scaling rather than a factor.
#
# Runs with the surface reference ON, so only the isotropic remainder is in play.
# val_search_lite keeps pixel-barrel hits only and skips the exact double solve,
# which is what makes a scan affordable at all.
set -e
cd /foo/matevz/mic-dev/current/src/standalone
unset DISPLAY
export LD_LIBRARY_PATH=.
N=${1:-10}
SAMPLE=/foo/matevz/mic-dev/trackingNtuple_HLT_2026_March.bin
T=../RecoTracker/MkFitCore/standalone/test

CMD=(./mkFit --geom CMS-phase2 --seed-input cmssw --input-file "$SAMPLE"
     --num-events "$N" --num-thr 1 --build-mimi --build-mimi-v2p2 --shell
     --shell-command 'gROOT->SetBatch(kTRUE)'
     --shell-command "gROOT->ProcessLine(\".L $T/val-prop.C\")"
     --shell-command 'val_seeds(0, true)'
     --shell-command 'val_search_lite(true)'
     --shell-command 'val_surf_q(true)')

for M in 1.0 1.5 2.0 3.0; do
  CMD+=(--shell-command "val_mat_scale($M)" --shell-command 'val_sr_reset()')
  for ((i=1;i<=N;i++)); do
    CMD+=(--shell-command "s.GoToEvent($i)" --shell-command 's.ProcessEventHlt()'
          --shell-command "val_sr_ev(s.event(), $i)")
  done
  CMD+=(--shell-command "val_sr_write(\"$T/val-mat-$M.root\")")
done
echo .q | "${CMD[@]}"
