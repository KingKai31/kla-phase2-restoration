#!/bin/bash
set -uo pipefail
cd /workspace/nffa_full

declare -A CHECKSUMS=(
  ["Biological.tar"]="c3a20755e2d937d70dd7d5be18f6006e"
  ["Fibres.tar"]="86fb065581eb3680f08ab2de9cea86eb"
  ["Films_Coated_Surface.tar"]="feb2b098a150c2ba6b6b35f2f18a6e7c"
  ["MEMS_devices_and_electrodes.tar"]="8bc4f788ec54de267a2b8cd0ac92066a"
  ["Nanowires.tar"]="005ffe16579c6352d3d521f58de56e8d"
  ["Particles.tar"]="6b73d88e65d3c00493076238c4230579"
  ["Patterned_surface.tar"]="19c0417faa8f977fdbf13d9c91ac3be8"
  ["Porous_Sponge.tar"]="851e6daf0e7ce45b68dbe9eb1338807e"
  ["Powder.tar"]="82da8aeb4d7a79c53bc3fefe9770e757"
  ["Tips.tar"]="0a85315213af1c32a14555d46e8f093a"
)

mkdir -p extracted
for name in "${!CHECKSUMS[@]}"; do
  expected="${CHECKSUMS[$name]}"
  echo "=== $name ==="
  if [ -f "extracted/.${name}.done" ]; then
    echo "  already done, skipping"
    continue
  fi
  success=0
  for attempt in 1 2 3; do
    if [ "$attempt" -gt 1 ]; then
      rm -f "$name"  # force a clean slate on a bash-level retry - never resume onto a
                      # file already proven wrong by a failed checksum on a prior attempt
    fi
    # -C - : resume within THIS curl invocation if the connection drops mid-transfer
    # (self-heals the truncation pattern this network path has shown for large files)
    # --retry/--retry-all-errors: curl's own internal retry-with-resume on network errors
    curl -sL -C - --retry 8 --retry-delay 5 --retry-all-errors --max-time 1800 \
         -o "$name" "https://b2share.eudat.eu/api/records/862nr-cn036/files/$name/content"
    actual=$(md5sum "$name" | awk '{print $1}')
    if [ "$actual" = "$expected" ]; then
      success=1
      break
    else
      echo "  attempt $attempt: checksum mismatch (expected $expected, got $actual), retrying"
    fi
  done
  if [ "$success" -ne 1 ]; then
    echo "  FAILED after 3 attempts - STOPPING"
    exit 1
  fi
  echo "  downloaded & verified OK"
  tar -xf "$name" -C extracted
  n_files=$(find "extracted/$(basename $name .tar)" -type f | wc -l)
  echo "  extracted: $n_files files"
  rm -f "$name"
  touch "extracted/.${name}.done"
  df -h /workspace | tail -1
done
echo "ALL_CATEGORIES_DONE"
