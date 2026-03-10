#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./download_lora_checkpoints.sh [--exp <exp_suffix>] <config_suffix> [checkpoint_id ...]

Examples:
  ./download_lora_checkpoints.sh --exp pi05_libero_skill_reason_lora_v3 pi05_libero_skill_reason_lora_v2 30000 40000 50000
  ./download_lora_checkpoints.sh pi05_libero_skill_reason_lora_v3 30000 40000 50000
  ./download_lora_checkpoints.sh pi05_libero_skill_reason_lora_v3

Notes:
  - <config_suffix> controls local config folder under checkpoints/<config_suffix>/...
  - If --exp is provided:
      repo_suffix = exp_suffix = <exp_suffix>
      config_suffix = positional <config_suffix>
  - If --exp is not provided:
      repo_suffix = config_suffix = exp_suffix = positional <config_suffix>
  - The Hugging Face repo is always n5zhong/<repo_suffix>.
  - If no checkpoint IDs are provided, the script downloads all files from the repo.
  - Also creates: assets/<config_suffix>/yilin-wu -> checkpoints/<config_suffix>/<exp_suffix>/<checkpoint>/assets/yilin-wu
    using the highest checkpoint ID available.
EOF
}

EXP_SUFFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)
            if [[ $# -lt 2 ]]; then
                echo "Error: --exp requires a value." >&2
                usage
                exit 1
            fi
            EXP_SUFFIX="$2"
            shift 2
            ;;
        --exp=*)
            EXP_SUFFIX="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: unknown option '$1'." >&2
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

CONFIG_SUFFIX="$1"
shift

if [[ -n "${EXP_SUFFIX}" ]]; then
    REPO_SUFFIX="${EXP_SUFFIX}"
else
    REPO_SUFFIX="${CONFIG_SUFFIX}"
    EXP_SUFFIX="${CONFIG_SUFFIX}"
fi

REPO_ID="n5zhong/${REPO_SUFFIX}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/" && pwd)
TARGET_DIR="${SCRIPT_DIR}/checkpoints/${CONFIG_SUFFIX}/${EXP_SUFFIX}"
ASSETS_DIR="${SCRIPT_DIR}/assets/${CONFIG_SUFFIX}"

create_assets_symlink() {
    local link_ckpt="$1"
    local link_name="${ASSETS_DIR}/yilin-wu"
    local link_target_rel="../../checkpoints/${CONFIG_SUFFIX}/${EXP_SUFFIX}/${link_ckpt}/assets/yilin-wu/"

    mkdir -p "${ASSETS_DIR}"

    if [[ -e "${link_name}" && ! -L "${link_name}" ]]; then
        echo "Error: ${link_name} exists and is not a symlink. Remove it and rerun." >&2
        exit 1
    fi

    if [[ ! -d "${TARGET_DIR}/${link_ckpt}/assets/yilin-wu" ]]; then
        echo "Error: checkpoint ${link_ckpt} does not contain assets/yilin-wu." >&2
        exit 1
    fi

    ln -sTfn "${link_target_rel}" "${link_name}"
    echo "Created symlink: ${link_name} -> ${link_target_rel}"
}

pick_highest_downloaded_checkpoint() {
    local best=""
    local ckpt=""
    for ckpt_dir in "${TARGET_DIR}"/*; do
        [[ -d "${ckpt_dir}" ]] || continue
        ckpt="$(basename "${ckpt_dir}")"
        [[ "${ckpt}" =~ ^[0-9]+$ ]] || continue
        [[ -d "${ckpt_dir}/assets/yilin-wu" ]] || continue
        if [[ -z "${best}" || "${ckpt}" -gt "${best}" ]]; then
            best="${ckpt}"
        fi
    done
    echo "${best}"
}

if command -v hf >/dev/null 2>&1; then
    HF_CMD=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD=(huggingface-cli)
else
    echo "Error: neither 'hf' nor 'huggingface-cli' is installed." >&2
    exit 1
fi

mkdir -p "${TARGET_DIR}"

if [[ $# -eq 0 ]]; then
    echo "No checkpoint IDs provided. Downloading all files from ${REPO_ID} into ${TARGET_DIR}"
    "${HF_CMD[@]}" download "${REPO_ID}" --repo-type model --local-dir "${TARGET_DIR}"
    LINK_CKPT="$(pick_highest_downloaded_checkpoint)"
    if [[ -z "${LINK_CKPT}" ]]; then
        echo "Error: no numeric checkpoint with assets/yilin-wu found in ${TARGET_DIR}." >&2
        exit 1
    fi
    create_assets_symlink "${LINK_CKPT}"
    exit 0
fi

INCLUDE_ARGS=()
HIGHEST_INPUT_CKPT=""
for ckpt in "$@"; do
    if [[ ! "${ckpt}" =~ ^[0-9]+$ ]]; then
        echo "Error: checkpoint ID must be numeric, got '${ckpt}'." >&2
        exit 1
    fi
    INCLUDE_ARGS+=(--include "${ckpt}/*")
    if [[ -z "${HIGHEST_INPUT_CKPT}" || "${ckpt}" -gt "${HIGHEST_INPUT_CKPT}" ]]; then
        HIGHEST_INPUT_CKPT="${ckpt}"
    fi
done

echo "Downloading checkpoints from ${REPO_ID} into ${TARGET_DIR}: $*"
"${HF_CMD[@]}" download "${REPO_ID}" --repo-type model --local-dir "${TARGET_DIR}" "${INCLUDE_ARGS[@]}"
create_assets_symlink "${HIGHEST_INPUT_CKPT}"
