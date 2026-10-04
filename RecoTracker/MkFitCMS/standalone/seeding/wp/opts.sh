#!/bin/bash
# Prints a working point's seedsurf options (wp/<WP>/seedsurf.opts) as one line, with every
# @file token replaced by that file's contents (the path relative to seeding/).
#
#   wp/opts.sh WP
set -e
D=$(cd "$(dirname "$0")/.." && pwd)
F=$D/wp/$1/seedsurf.opts
[ -f "$F" ] || { echo "wp/opts.sh: no working point '$1' ($F)" >&2; exit 1; }
for t in $(cat "$F"); do
  case $t in
    @*) cat "$D/${t#@}" ;;
    *) printf '%s\n' "$t" ;;
  esac
done | tr '\n' ' '
echo
