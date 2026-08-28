#!/usr/bin/env bash
# Run the complete ep_test.py sanity matrix inside one Slurm allocation.

set -uo pipefail

: "${NP:=8}"
: "${NCCL_EP_PYTHON_TEST_TOKENS:=16}"
: "${NCCL_EP_PYTHON_TEST_HIDDEN:=1024}"
: "${NCCL_EP_PYTHON_TEST_TIMEOUT:=300}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
EP_TEST_PY="${EP_TEST_PY:-${REPO_ROOT}/nccl_ep/ep_test.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${EP_TEST_PY}" ]]; then
  echo "ERROR: ep_test.py not found: ${EP_TEST_PY}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! command -v srun >/dev/null 2>&1; then
  echo "ERROR: srun is required; run this script inside a Slurm job" >&2
  exit 1
fi

passed=0
failed=0
failed_cases=()

run_case() {
  local name="$1"
  shift

  echo
  echo "================================================================"
  echo "CASE: ${name}"
  echo "ARGS: $*"
  echo "================================================================"

  if timeout "${NCCL_EP_PYTHON_TEST_TIMEOUT}" \
      srun \
        -N 1 \
        -n "${NP}" \
        --exclusive \
        --mpi=pmix \
        --cpu-bind=none \
        --export=ALL \
        "${PYTHON_BIN}" "${EP_TEST_PY}" \
        -t "${NCCL_EP_PYTHON_TEST_TOKENS}" \
        -d "${NCCL_EP_PYTHON_TEST_HIDDEN}" \
        "$@"; then
    ((passed += 1))
    echo "CASE_RESULT: PASS (${name})"
  else
    local rc=$?
    ((failed += 1))
    failed_cases+=("${name} (rc=${rc})")
    echo "CASE_RESULT: FAIL (${name}, rc=${rc})"
  fi
}

# Baseline algorithm/layout coverage.
run_case "ll-em" -a ll -L em
run_case "ll-rm" -a ll -L rm
run_case "ht-fl" -a ht -L fl
run_case "ht-em" -a ht -L em

# Quantization smoke coverage for both algorithms.
run_case "ll-quant" -a ll -Q
run_case "ht-quant" -a ht -Q

# LL staged execution: dispatch-only, combine-only, and both stages.
run_case "ll-em-staged-dispatch" -a ll -L em -s dispatch
run_case "ll-em-staged-combine" -a ll -L em -s combine
run_case "ll-em-staged-both" -a ll -L em -s both
run_case "ll-rm-staged-both" -a ll -L rm -s both

# Handle reuse and routing refresh.
run_case "ht-fl-cached" -a ht -L fl -c
run_case "ll-em-update" -a ll -L em --update
run_case "ll-rm-update" -a ll -L rm --update
run_case "ht-fl-update" -a ht -L fl --update
run_case "ht-em-update" -a ht -L em --update

# HT allocation, pass-direction, and overflow modes.
run_case "ht-fl-eager" -a ht -L fl -q
run_case "ht-em-eager" -a ht -L em -q
run_case "ht-fl-backward" -a ht -L fl --backward
run_case "ht-em-backward" -a ht -L em --backward
run_case "ht-fl-overflow-drop" -a ht -L fl --overflow-drop

# Metadata contracts.
run_case "ll-rm-local-expert-ids" -a ll -L rm --expert-id-kind local
run_case "ll-rm-global-expert-ids" -a ll -L rm --expert-id-kind global
run_case "ht-fl-local-expert-ids" -a ht -L fl --expert-id-kind local
run_case "ht-fl-global-expert-ids" -a ht -L fl --expert-id-kind global
run_case "ht-em-alignment" -a ht -L em --alignment 8

# LL active-mask lifecycle and random-routing smoke coverage.
run_case "ll-em-mask" -a ll -L em --mask
run_case "ll-em-random" -a ll -L em -r
run_case "ht-fl-random" -a ht -L fl -r

echo
echo "================================================================"
echo "NCCL EP PYTHON TEST SUMMARY"
echo "passed=${passed} failed=${failed} total=$((passed + failed))"
if ((failed > 0)); then
  printf 'failed_case=%s\n' "${failed_cases[@]}"
  echo "RESULT=FAIL"
  exit 1
fi
echo "RESULT=PASS"
