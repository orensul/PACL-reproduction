#!/usr/bin/env bash
# Download MS-COCO 2017 into the layout train_pacl.py expects.
#   usage: bash scripts/download_coco.sh /path/to/MSCOCO
# Result:
#   $ROOT/train2017/            (~18 GB, 118k images)
#   $ROOT/val2017/              (~1 GB,  5k images)
#   $ROOT/annotations/captions_train2017.json
#   $ROOT/annotations/captions_val2017.json
set -euo pipefail

ROOT="${1:-/home/Dataset/Visual_Recognition/MSCOCO}"
mkdir -p "$ROOT"
cd "$ROOT"

echo "Downloading MS-COCO 2017 into $ROOT ..."
for f in train2017.zip val2017.zip annotations_trainval2017.zip; do
  if [ ! -f "$f" ]; then
    wget -c "http://images.cocodataset.org/zips/$f" 2>/dev/null \
      || wget -c "http://images.cocodataset.org/annotations/$f"
  fi
done

echo "Unzipping ..."
for f in train2017.zip val2017.zip annotations_trainval2017.zip; do
  unzip -n -q "$f"
done

echo "Done. Expected paths:"
echo "  $ROOT/train2017/  $ROOT/val2017/  $ROOT/annotations/captions_{train,val}2017.json"
