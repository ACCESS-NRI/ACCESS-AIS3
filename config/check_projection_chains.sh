#!/bin/bash
# Health check for running SSA projection chains (worklog 6.35). For every unfinished chunk directory in
# execution_SSA (no lock file, not archived), report:
#   DEAD    the inner job has ended (PBS footer in its outlog) without a lock file -- a driver started before the
#           watched-solve fix (2026-11-21 ~22:45) will idle on it until its wait budget runs out;
#   STALL   its job is running but the outbin has not grown for more than STALL_MIN minutes (default 90; collapse
#           sub-solves STALL_MIN_COLLAPSE, default 300: they are 1 model year and write only at the start and end);
#   ok      progressing (outbin size and minutes since last write shown).
# usage: bash check_projection_chains.sh [pattern, e.g. hm4000]   (from anywhere; read only)
ROOT=/g/data/au88/jh7060/ACCESS-AIS3
STALL_MIN=${STALL_MIN:-90}
STALL_MIN_COLLAPSE=${STALL_MIN_COLLAPSE:-300}
pat=${1:-.}
now=$(date +%s)
cd "$ROOT/execution_SSA" || exit 1
for d in $(ls -d AIS3_proj_ssa_*_[0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]* 2>/dev/null \
           | grep -v -E "_failed|_stalled|_attempts|_killed|_walltime|smoke" | grep -E "$pat"); do
    [ -f "$d/$d.lock" ] && continue
    o="$d/$d.outlog"; f="$d/$d.outbin"
    job=$(qselect -u "$USER" -N "$d.queue" 2>/dev/null | head -1)
    if grep -a -q "Exit Status:" "$o" 2>/dev/null; then
        printf "DEAD   %-85s %s\n" "${d#AIS3_proj_ssa_}" "$(grep -a 'Exit Status:' "$o" | tail -1 | tr -s ' ')"
    elif [ -n "$job" ] && [ -f "$f" ]; then
        age=$(( (now - $(stat -c %Y "$f")) / 60 )); mb=$(( $(stat -c %s "$f") / 1000000 ))
        lim=$STALL_MIN; case $d in *_collapse_*) lim=$STALL_MIN_COLLAPSE;; esac
        if [ "$age" -gt "$lim" ]; then st=STALL; else st=ok; fi
        printf "%-6s %-85s %6d MB, last write %4d min ago (job %s)\n" "$st" "${d#AIS3_proj_ssa_}" "$mb" "$age" "$job"
    elif [ -n "$job" ]; then
        printf "ok     %-85s starting or queued (job %s)\n" "${d#AIS3_proj_ssa_}" "$job"
    fi
done
