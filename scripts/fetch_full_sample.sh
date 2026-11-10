#!/bin/bash
# first and last episode per batch -> data/full/<batch>/<ep>.mp4
ROOT=$1; OUT=$2
B=https://huggingface.co/datasets/surgicalrobotics/appendectomy-invivo/resolve/main
cd "$ROOT" && for b in batch_*/; do ls -d $b*/ | sed -n '1p;$p'; done | sed 's#/$##' | sort -u | \
xargs -P 6 -I{} sh -c 'o="'"$OUT"'/{}.mp4"; mkdir -p "$(dirname "$o")"; [ -s "$o" ] || curl -sfL --retry 5 --retry-delay 20 -o "$o.part" "'"$B"'/{}/camera.mp4" && mv "$o.part" "$o"'
echo done; find "$OUT" -name "*.mp4" | wc -l
