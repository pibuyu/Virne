#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/plots/run_all_plots.sh --run-dir <RUN_DIR> [options]
  scripts/plots/run_all_plots.sh --training-csv <CSV> --records-csv <CSV> [options]

Options:
  --run-dir <DIR>         Run directory containing logs/ and records/
  --training-csv <PATH>   Explicit training_info.csv path
  --records-csv <PATH>    Explicit records csv path
  --out-dir <DIR>         Output directory for pngs (default: alongside CSV)
  --x <COL>               X-axis column name override
  --only-enter            For records plots, filter to event_type==1
  --window <N>            Rolling mean window for v_net_r2c_ratio
  -h, --help              Show help
EOF
}

RUN_DIR=""
TRAINING_CSV=""
RECORDS_CSV=""
OUT_DIR=""
X_COL=""
ONLY_ENTER=0
WINDOW=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"; shift 2;;
    --training-csv)
      TRAINING_CSV="$2"; shift 2;;
    --records-csv)
      RECORDS_CSV="$2"; shift 2;;
    --out-dir)
      OUT_DIR="$2"; shift 2;;
    --x)
      X_COL="$2"; shift 2;;
    --only-enter)
      ONLY_ENTER=1; shift;;
    --window)
      WINDOW="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1;;
  esac
done

if [[ -z "$RUN_DIR" && -z "$TRAINING_CSV" && -z "$RECORDS_CSV" ]]; then
  echo "Error: provide --run-dir or explicit CSV paths." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAINING_ARGS=()
RECORDS_ARGS=()

if [[ -n "$RUN_DIR" ]]; then
  TRAINING_ARGS+=(--run-dir "$RUN_DIR")
  RECORDS_ARGS+=(--run-dir "$RUN_DIR")
fi
if [[ -n "$TRAINING_CSV" ]]; then
  TRAINING_ARGS+=(--training-csv "$TRAINING_CSV")
fi
if [[ -n "$RECORDS_CSV" ]]; then
  RECORDS_ARGS+=(--records-csv "$RECORDS_CSV")
fi
if [[ -n "$X_COL" ]]; then
  TRAINING_ARGS+=(--x "$X_COL")
  RECORDS_ARGS+=(--x "$X_COL")
fi
if [[ "$ONLY_ENTER" -eq 1 ]]; then
  RECORDS_ARGS+=(--only-enter)
fi

if [[ -n "$OUT_DIR" ]]; then
  mkdir -p "$OUT_DIR"
fi

FAIL=0
run_cmd() {
  local name="$1"; shift
  echo "[plot] $name"
  if "$@"; then
    :
  else
    echo "[plot] failed: $name" >&2
    FAIL=1
  fi
}

out_arg() {
  local slug="$1"
  if [[ -n "$OUT_DIR" ]]; then
    echo "--out" "${OUT_DIR}/${slug}.png"
  fi
}

# training plots
run_cmd "training_loss" python "${SCRIPT_DIR}/plot_training_loss.py" "${TRAINING_ARGS[@]}" $(out_arg training_loss)
run_cmd "training_reward" python "${SCRIPT_DIR}/plot_training_reward.py" "${TRAINING_ARGS[@]}" $(out_arg training_reward)
run_cmd "training_actor_loss" python "${SCRIPT_DIR}/plot_training_actor_loss.py" "${TRAINING_ARGS[@]}" $(out_arg training_actor_loss)
run_cmd "training_critic_loss" python "${SCRIPT_DIR}/plot_training_critic_loss.py" "${TRAINING_ARGS[@]}" $(out_arg training_critic_loss)
run_cmd "training_entropy_loss" python "${SCRIPT_DIR}/plot_training_entropy_loss.py" "${TRAINING_ARGS[@]}" $(out_arg training_entropy_loss)
run_cmd "training_approx_kl" python "${SCRIPT_DIR}/plot_training_approx_kl.py" "${TRAINING_ARGS[@]}" $(out_arg training_approx_kl)

# records plots
run_cmd "acceptance_rate" python "${SCRIPT_DIR}/plot_acceptance_rate.py" "${RECORDS_ARGS[@]}" $(out_arg acceptance_rate)
run_cmd "long_term_r2c" python "${SCRIPT_DIR}/plot_long_term_r2c.py" "${RECORDS_ARGS[@]}" $(out_arg long_term_r2c)
run_cmd "long_term_time_r2c" python "${SCRIPT_DIR}/plot_long_term_time_r2c.py" "${RECORDS_ARGS[@]}" $(out_arg long_term_time_r2c)
run_cmd "p_net_node_resource_utilization" python "${SCRIPT_DIR}/plot_p_net_node_utilization.py" "${RECORDS_ARGS[@]}" $(out_arg p_net_node_resource_utilization)
run_cmd "p_net_link_resource_utilization" python "${SCRIPT_DIR}/plot_p_net_link_utilization.py" "${RECORDS_ARGS[@]}" $(out_arg p_net_link_resource_utilization)
run_cmd "inservice_count" python "${SCRIPT_DIR}/plot_inservice_count.py" "${RECORDS_ARGS[@]}" $(out_arg inservice_count)

VNET_ARGS=()
if [[ -n "$WINDOW" ]]; then
  VNET_ARGS+=(--window "$WINDOW")
fi
run_cmd "v_net_r2c_ratio" python "${SCRIPT_DIR}/plot_v_net_r2c_ratio.py" "${RECORDS_ARGS[@]}" "${VNET_ARGS[@]}" $(out_arg v_net_r2c_ratio)

exit "$FAIL"
