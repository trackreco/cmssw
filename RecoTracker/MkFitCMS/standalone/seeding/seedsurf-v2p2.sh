#!/bin/bash
# Seeding for v2p2 at a named working point: seedsurf-chain.sh plus the options of wp/<WP>/seedsurf.opts.
#
#   [WP=2026-10-04] seedsurf-v2p2.sh [seedsurf options ...]      same environment as seedsurf-chain.sh
#
# WP: a directory under wp/ (default 2026-10-02, the working point agreed with the maintainer). Each
# working point keeps the finder settings it was measured with beside its seedsurf options; wp/README.md
# lists them, with their measurements. seedsurf-chain.sh stays the reference for the identity checks of
# seed-ref/.
D=$(cd "$(dirname "$0")" && pwd)
OPTS=$("$D"/wp/opts.sh "${WP:-2026-10-02}") || exit 1
# $OPTS is deliberately unquoted: one option word per token
exec "$D"/seedsurf-chain.sh $OPTS "$@"
