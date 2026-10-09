#!/bin/bash
# Seeding for v2p2 at a named working point: seedsurf-chain.sh plus the options of wp/<WP>/seedsurf.opts.
#
#   [WP=2026-10-02] seedsurf-v2p2.sh [seedsurf options ...]      same environment as seedsurf-chain.sh
#
# WP: a directory under wp/ (default 2026-10-06-transwin, the working point of CMSSW's
# mkfit-phase2-seeder.json; 2026-10-04 and 2026-10-02 before). Each working point keeps the finder settings
# it was measured with beside its seedsurf options; wp/README.md lists them, with their measurements.
# seedsurf-chain.sh stays the reference for the identity checks of
# seed-ref/.
D=$(cd "$(dirname "$0")" && pwd)
OPTS=$("$D"/wp/opts.sh "${WP:-2026-10-06-transwin}") || exit 1
# $OPTS is deliberately unquoted: one option word per token
exec "$D"/seedsurf-chain.sh $OPTS "$@"
