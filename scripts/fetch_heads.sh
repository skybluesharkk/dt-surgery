#!/bin/bash
# usage: fetch_heads.sh <data_root> <out_dir> <bytes>
ROOT=$1; OUT=$2; N=$3
B=https://huggingface.co/datasets/surgicalrobotics/appendectomy-invivo/resolve/main
cd "$ROOT" && ls -d batch_*/*/ | sed 's#/$##' | while read p; do echo "$p"; done | \
xargs -P 8 -I{} sh -c 'o="'"$OUT"'/{}.mp4"; mkdir -p "$(dirname "$o")"; [ -s "$o" ] || curl -sfL -r 0-'"$N"' -o "$o" "'"$B"'/{}/camera.mp4"'
