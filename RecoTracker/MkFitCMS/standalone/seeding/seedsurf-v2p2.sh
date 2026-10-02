#!/bin/bash
# The working point for seeding v2p2 (2026-10-02): seedsurf-chain.sh plus one missed pixel hit allowed
# on the way to OT1-P (--chain 1 --chain-holes-ot 1 --chain-inner-ot-only, windows-D121/ot1p-hole.txt),
# d0_max 0.5 mm, the beam region +-15 cm in z (3.5 sigma of the D121 beam spot), and the fake score
# S < 0.5 where the candidate's line has |eta| >= 1.7 (S < 0.75 below).
#
#   seedsurf-v2p2.sh [seedsurf options ...]      same environment as seedsurf-chain.sh
#
# Measured with v2p2 and CMSSW's loose selection on 30 PU200 events (cmssw_20_mkseed/SESSIONS.md S20):
# 82.74 % efficiency, 4675 fakes, 191 ms/ev of seeding on phi3. seedsurf-chain.sh stays the reference
# for the identity checks of seed-ref/.
D=$(cd "$(dirname "$0")" && pwd)
exec "$D"/seedsurf-chain.sh --chain-batch --fk-score 0.75 --fk-score-fwd 0.5 1.7 --fk-ot2 1.5 --fk-shape \
  $(cat "$D"/windows-D121/ot1p-hole.txt) --chain 1 --chain-holes-ot 1 --chain-inner-ot-only \
  --d0-max 0.05 --zv 15 "$@"
