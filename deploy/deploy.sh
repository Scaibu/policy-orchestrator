#!/usr/bin/env bash
set -euo pipefail

get_root_dir() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/.." && pwd
}

prepare_public_dir() {
  local public_dir="$1"
  mkdir -p "$public_dir"
}

sync_core_assets() {
  local src_dir="$1"
  local dest_dir="$2"
  cp -f "$src_dir/index.html" "$dest_dir/index.html"
  cp -f "$src_dir/app.js" "$dest_dir/app.js"
  cp -f "$src_dir/styles.css" "$dest_dir/styles.css"
  cp -f "$src_dir/proofs.js" "$dest_dir/proofs.js"
  cp -f "$src_dir/contracts.js" "$dest_dir/contracts.js"
  if [ -f "$src_dir/logo.svg" ]; then
    cp -f "$src_dir/logo.svg" "$dest_dir/logo.svg"
  fi
  if [ -f "$src_dir/robots.txt" ]; then
    cp -f "$src_dir/robots.txt" "$dest_dir/robots.txt"
  fi
  if [ -f "$src_dir/sitemap.xml" ]; then
    cp -f "$src_dir/sitemap.xml" "$dest_dir/sitemap.xml"
  fi
  for vfile in "$src_dir"/google*.html; do
    [ -f "$vfile" ] && cp -f "$vfile" "$dest_dir/"
  done
  if [ -d "$src_dir/diagrams" ]; then
    mkdir -p "$dest_dir/diagrams"
    cp -rf "$src_dir/diagrams/"* "$dest_dir/diagrams/"
  fi
}

sync_domain_specs() {
  local root_src="$1"
  local dest_dir="$2"
  for domain in "$root_src"/*/; do
    [ -d "$domain" ] || continue
    local name
    name="$(basename "$domain")"
    if [ "$name" = "visualizer" ] || [ "$name" = "node_modules" ] || [ "$name" = "__pycache__" ]; then
      continue
    fi
    mkdir -p "$dest_dir/$name"
    for algo in "$domain"/*/; do
      [ -d "$algo" ] || continue
      local algo_name
      algo_name="$(basename "$algo")"
      local target_dir="$dest_dir/$name/$algo_name"
      mkdir -p "$target_dir"
      if [ -f "$algo/math.tex" ]; then
        cp -f "$algo/math.tex" "$target_dir/math.tex"
      fi
      if [ -f "$algo/math.pdf" ]; then
        cp -f "$algo/math.pdf" "$target_dir/math.pdf"
      fi
    done
  done
}

ensure_firebase_cli() {
  if ! command -v firebase &>/dev/null; then
    local node_path
    node_path="$HOME/.nvm/versions/node/$(node -v 2>/dev/null || echo '')/bin"
    if [ -d "$node_path" ]; then
      export PATH="$PATH:$node_path"
    fi
  fi
  if ! command -v firebase &>/dev/null; then
    echo "firebase CLI not found in PATH" >&2
    exit 1
  fi
}

execute_deploy() {
  local firebase_dir="$1"
  cd "$firebase_dir"
  rm -rf .firebase
  firebase deploy --only hosting --project nn-visualizer
}

main() {
  local root_dir
  root_dir="$(get_root_dir)"
  local base_src="$root_dir/src/features/code_engine/algos/nn"
  local visualizer_src="$base_src/visualizer"
  local deploy_dir="$root_dir/deploy/firebase"
  local public_dest="$deploy_dir/public"

  ensure_firebase_cli
  prepare_public_dir "$public_dest"
  sync_core_assets "$visualizer_src" "$public_dest"
  sync_domain_specs "$base_src" "$public_dest"
  execute_deploy "$deploy_dir"
}

main "$@"
