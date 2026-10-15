#!/bin/bash
# run_cupy_compare.sh <npes> <outdir> [program ...]
# Runs the ports with CNUM_REFERENCE=cupynumeric: charmnumeric +p<npes> vs cuPyNumeric on <npes> GPUs.
# Env: TREE (charmtyles tree, default main), BUILD (default build-gpu), EX_TIMEOUT.
# Results: <outdir>/results.jsonl, per-program <name>.client.log / .server.log.
NP=$1; OUT=$(realpath -m $2); shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
export BUILD=${BUILD:-build-gpu}
. /u/atahilyani/cnum-gpu/harness/env_tree.sh
mkdir -p $OUT; B=$OUT/bin; mkdir -p $B
# snapshot so rebuilding the tree mid-run doesn't kill the server
cp $CN/src/$BUILD/{server.out,charmrun} $B/
export CNUM_REFERENCE=cupynumeric CNUM_CUPY_GPUS=$NP CNUM_REF_JSON=$OUT/results.jsonl LOGDIR=$OUT
export EX_TIMEOUT=${EX_TIMEOUT:-900}
[ $# -eq 0 ] && set -- $(cd $HERE && ls *.py | grep -v "^_" | sed 's/\.py$//')
PORT=$((22000 + RANDOM % 5000))
for ex in "$@"; do
  PORT=$((PORT+1))
  /u/atahilyani/cnum-gpu/harness/run_one.sh $B/server.out $NP $PORT $HERE/$ex.py
done 2>&1 | tee $OUT/summary.txt
rm -rf $B
