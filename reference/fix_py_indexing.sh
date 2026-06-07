#!/usr/bin/env bash
# Fix indexing of .py files in reference (and subdirs): ensure 1_name.py, 2_name.py, ...
# with no duplicate indices. Processes each directory separately.
# Usage: ./fix_py_indexing.sh [REF_DIR] [--dry-run]
#   REF_DIR defaults to the directory containing this script.
#   Run with --dry-run first to see planned renames. Files are sorted by
#   current index then name; new indices are 1, 2, 3, ... per directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${1:-$SCRIPT_DIR}"
DRY_RUN=false
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=true

# Find all directories that contain at least one numbered .py file (N_name.py)
get_dirs() {
  find "$REF_DIR" -type f -name "*.py" -print0 | while IFS= read -r -d '' f; do
    base=$(basename "$f")
    if [[ "$base" =~ ^[0-9]+_.+\.py$ ]]; then
      dirname "$f"
    fi
  done | sort -u
}

sort_key() {
  local path="$1"
  local base="${path##*/}"
  if [[ "$base" =~ ^([0-9]+)_(.+)$ ]]; then
    printf "%05d_%s" "$((10#${BASH_REMATCH[1]}))" "${BASH_REMATCH[2]}"
  fi
}

temp_prefix="__reindex_"

while IFS= read -r dir; do
  [[ -d "$dir" ]] || continue

  files=()
  for f in "$dir"/*.py; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f")
    [[ "$base" =~ ^[0-9]+_.+\.py$ ]] || continue
    files+=("$f")
  done

  (( ${#files[@]} == 0 )) && continue

  # Sort by (numeric prefix, then name) for deterministic order
  sorted_files=()
  while IFS= read -r path; do
    [[ -n "$path" ]] && sorted_files+=("$path")
  done < <(for f in "${files[@]}"; do echo "$(sort_key "$f")|$f"; done | sort -t'|' -k1,1 | cut -d'|' -f2-)

  # Two-phase rename: first to temp names (avoid overwriting), then to final
  i=1
  for path in "${sorted_files[@]}"; do
    base=$(basename "$path")
    [[ "$base" =~ ^[0-9]+_(.+)$ ]] || continue
    name="${BASH_REMATCH[1]}"
    new_base="${i}_${name}"
    if [[ "$base" == "$new_base" ]]; then
      (( i++ )) || true
      continue
    fi
    temp_name="${temp_prefix}${i}_${name}"
    if "$DRY_RUN"; then
      echo "would rename: $path -> $dir/$new_base (via $temp_name)"
    else
      mv "$path" "$dir/$temp_name"
    fi
    (( i++ )) || true
  done

  if "$DRY_RUN"; then
    continue
  fi

  for temp in "$dir"/${temp_prefix}*.py; do
    [[ -f "$temp" ]] || continue
    new_base=$(basename "$temp" | sed "s/^${temp_prefix}//")
    mv "$temp" "$dir/$new_base"
    echo "  $dir: ... -> $new_base"
  done
done < <(get_dirs)

echo "Done."
