#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
PIP_BIN="${PIP_BIN:-${VENV_DIR}/bin/pip}"

ensure_virtualenv() {
  if [[ -d "${VENV_DIR}" && -x "${PYTHON_BIN}" ]]; then
    return
  fi

  python -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install --upgrade pip
}

run_python() {
  ensure_virtualenv
  "${PYTHON_BIN}" "$@"
}

run_pip() {
  ensure_virtualenv
  "${PIP_BIN}" "$@"
}

run_with_coverage() {
  local module="$1"
  shift
  run_python -m coverage run --source="src/orchestrator" -m "$module" "$@"
}
