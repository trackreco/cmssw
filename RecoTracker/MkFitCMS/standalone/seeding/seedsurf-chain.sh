#!/bin/bash
# The chosen seedsurf configuration (2026-09-27): the feed-forward chain,
# starting up to two crossed layers late, no other holes, cleaning N = 3,
# with the D121 window tables in windows-D121/.
#
#   seedsurf-chain.sh [seedsurf options ...]
#
# Environment: B (the standalone build directory, holding seedsurf; default the current
# directory, as for mkFit), SS (the binary in B, default seedsurf, for an A/B),
# S (the sample), GEOM, BIND (truth binding in cm; needs SimHitStates in the
# sample, empty to turn it off). Anything on the command line is appended, e.g.
#
#   seedsurf-chain.sh --first-event 40 --num-events 60 --truth truth.txt
#
# gives the chain row of the research README (1431.5 found tracks / ev, 27.5k quads, fake 0.343),
# frozen since 2026-10-01 in mkFit-external's mkfit-standalone-attic/mkfit-seeding/README.md.

set -e
D=$(cd "$(dirname "$0")" && pwd)
B=${B:-$PWD}
[ -x "$B/${SS:-seedsurf}" ] || { echo "seedsurf-chain.sh: no ${SS:-seedsurf} in $B (set B to the build directory)" >&2; exit 1; }
S=${S:-/foo/matevz/mic-dev/ttbar-PU200-D121-C22-100ev.bin}
GEOM=${GEOM:-CMS-phase2-Run4D121}
BIND=${BIND-0.05}

WIN=$(cat "$D"/windows-D121/pixel.txt "$D"/windows-D121/ot1p.txt "$D"/windows-D121/skip.txt "$D"/windows-D121/shape.txt)

cd "$B"
# $WIN is deliberately unquoted: one option word per token
LD_LIBRARY_PATH=. exec ./"${SS:-seedsurf}" --input-file "$S" --geom "$GEOM" \
  --pt-min 0.9 --marg-b 0.001 ${BIND:+--bind $BIND} \
  $WIN \
  --chain 0 --chain-start-holes 2 --chain-lead-only --dedup 3 \
  "$@"
