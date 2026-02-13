#!/usr/bin/env bash
set -euo pipefail

echo "INFO: scripts/run_four_workers_shared.sh is deprecated. Using run_six_workers_shared.sh." >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_six_workers_shared.sh" "$@"
