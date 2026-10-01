#!/usr/bin/env bash
# Helper for the TPU v6e-1 VM used by this repo.
#   scripts/tpu_vm.sh create [spot|ondemand]   try zones in order until one has capacity
#   scripts/tpu_vm.sh setup                    install uv, sync deps, download data, enable THP
#   scripts/tpu_vm.sh push                     copy the working tree (tracked + untracked, minus ignored) to the VM
#   scripts/tpu_vm.sh ssh [command]            run a command in ~/speedrun on the VM
#   scripts/tpu_vm.sh pull-logs                copy ~/speedrun/logs/*.txt back to ./logs/
#   scripts/tpu_vm.sh status | delete
set -euo pipefail

PROJECT="${TPU_PROJECT:-tpuproject-510313}"
NAME="${TPU_NAME:-speedrun-v6e1}"
VERSION="${TPU_VERSION:-v6e-ubuntu-2404}"
ZONES="${TPU_ZONES:-us-east1-d us-central1-a us-central1-b us-central1-c us-west1-c us-south1-a us-south1-c asia-southeast1-b us-east5-a us-east5-b}"
ZONE_FILE="$(dirname "$0")/../.tpu_zone"
REMOTE_DIR="speedrun"

zone() {
    if [[ -n "${TPU_ZONE:-}" ]]; then echo "$TPU_ZONE"; elif [[ -f "$ZONE_FILE" ]]; then cat "$ZONE_FILE"; else
        echo "no zone: run create first or set TPU_ZONE" >&2; exit 1; fi
}

gssh() {
    gcloud compute tpus tpu-vm ssh "$NAME" --zone="$(zone)" --project="$PROJECT" --quiet --command="$1"
}

case "${1:-}" in
create)
    mode="${2:-spot}"
    flags=()
    [[ "$mode" == "spot" ]] && flags+=(--spot)
    for z in $ZONES; do
        echo "=== $z ($mode)"
        if gcloud compute tpus tpu-vm create "$NAME" --zone="$z" --project="$PROJECT" \
            --accelerator-type=v6e-1 --version="$VERSION" "${flags[@]}" 2>&1 | grep -E "Created|ERROR|message"; then :; fi
        if [[ "$(gcloud compute tpus tpu-vm describe "$NAME" --zone="$z" --project="$PROJECT" \
            --format='value(state)' 2>/dev/null)" == "READY" ]]; then
            echo "$z" > "$ZONE_FILE"
            echo "READY in $z"
            exit 0
        fi
        gcloud compute tpus tpu-vm delete "$NAME" --zone="$z" --project="$PROJECT" --quiet >/dev/null 2>&1 || true
    done
    echo "no capacity in any zone" >&2
    exit 1
    ;;
setup)
    gssh "sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'; \
        (command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh) >/dev/null 2>&1; \
        mkdir -p ~/$REMOTE_DIR"
    "$0" push
    gssh "cd ~/$REMOTE_DIR && ~/.local/bin/uv sync -q && .venv/bin/python data/cached_fineweb10B.py 10 && du -sh data/fineweb10B"
    ;;
push)
    tmp="$(mktemp -d)"
    git ls-files --cached --others --exclude-standard -z | grep -zv '^third-party/' | tar --null -T - -czf "$tmp/src.tgz"
    gcloud compute tpus tpu-vm scp "$tmp/src.tgz" "$NAME":"~/$REMOTE_DIR/src.tgz" --zone="$(zone)" --project="$PROJECT" --quiet
    gssh "cd ~/$REMOTE_DIR && tar xzf src.tgz && rm src.tgz"
    rm -rf "$tmp"
    ;;
ssh)
    shift
    gssh "cd ~/$REMOTE_DIR && ${*:-bash -l}"
    ;;
pull-logs)
    mkdir -p logs
    gcloud compute tpus tpu-vm scp "$NAME":"~/$REMOTE_DIR/logs/*.txt" logs/ --zone="$(zone)" --project="$PROJECT" --quiet
    ;;
status)
    gcloud compute tpus tpu-vm describe "$NAME" --zone="$(zone)" --project="$PROJECT" --format="value(state,health)"
    ;;
delete)
    gcloud compute tpus tpu-vm delete "$NAME" --zone="$(zone)" --project="$PROJECT" --quiet
    rm -f "$ZONE_FILE"
    ;;
*)
    sed -n '2,9p' "$0"
    exit 1
    ;;
esac
