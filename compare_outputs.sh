#!/bin/sh
# compare_outputs.sh REF_DIR NEW_DIR REPORT
# Compares the outputs of a run on another machine (NEW_DIR) with the reference outputs (REF_DIR), file by file.
# The files compared are the ones that FALLBACK_COMPARISON.txt of the reference run lists, that is, the gas and
# security programs whose output must not depend on the build. rowmacc_scaling.txt contains a timing table, so
# only its gas tables and check lines are compared. Exit status 1 if a file differs or is missing.
REF=$1; NEW=$2; REP=$3
gasof() { sed -n '/^TOTAL GAS/,/^TIME/p' "$1"; grep -E 'bit-exact|harness|CHECK' "$1"; }
{
  echo "Outputs in $NEW compared with the reference outputs in $REF, file by file (cmp)"
  echo "machine: $(uname -m)"
  for b in $(awk '/^identical/ {print $2}' "$REF/FALLBACK_COMPARISON.txt"); do
    if [ ! -f "$NEW/$b" ]; then echo "MISSING     $b"; continue; fi
    if [ "$b" = "rowmacc_scaling.txt" ]; then
      if [ "$(gasof "$REF/$b")" = "$(gasof "$NEW/$b")" ]; then echo "identical   $b (gas tables and check lines, timing excluded)"; else echo "DIFFERENT   $b (gas tables or check lines)"; fi
    elif cmp -s "$REF/$b" "$NEW/$b"; then echo "identical   $b"; else echo "DIFFERENT   $b"; fi
  done
} > "$REP"
I=$(grep -c '^identical' "$REP"); D=$(grep -c '^DIFFERENT' "$REP"); M=$(grep -c '^MISSING' "$REP")
echo "summary: $I identical, $D different, $M missing" >> "$REP"
cat "$REP"
[ "$D" = "0" ] && [ "$M" = "0" ] && [ "$I" != "0" ]
