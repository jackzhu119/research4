#!/usr/bin/env bash
# Independent B host only. This script neither assumes inherited GitHub rights
# nor modifies the old host-A ledger, datasets, spectra, or historical models.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_research4_worker_b_20261106.sh --run-id ID [options]
  --detach         Start the bounded supervisor in a terminal-independent session.
  --check-only     Read-only checks: no installation, checkout, quantum job or push.
  --smoke-test     Water-only qualification; not a medicinal-plant result.
  --no-upload      Explicit local test; automatic Git return is disabled.
  --python PATH   Python 3.12 preferred; 3.11 is accepted subject to real install.
  --venv PATH     New, dedicated managed environment (default .runtime/worker-b-20261106/venv).
  --attempt ID    Fresh supervisor invocation ID (default launch-UTC timestamp).
EOF
}

fail() { printf '%s\n' "Worker B blocked: $*" >&2; exit 2; }

run_id=''; detach=0; check_only=0; smoke_test=0; no_upload=0
python_command='python3'; venv_relative='.runtime/worker-b-20261106/venv'
attempt="launch-$(date -u +%Y%m%dT%H%M%SZ)"
while (($#)); do
  case "$1" in
    --run-id|--python|--venv|--attempt)
      (($# >= 2)) || fail "missing value for $1"
      case "$1" in
        --run-id) run_id=$2;; --python) python_command=$2;;
        --venv) venv_relative=$2;; --attempt) attempt=$2;;
      esac
      shift 2;;
    --detach) detach=1; shift;;
    --check-only) check_only=1; shift;;
    --smoke-test) smoke_test=1; shift;;
    --no-upload) no_upload=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument $1";;
  esac
done
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,59}$ ]] || fail "--run-id needs 1–60 safe ASCII letters/digits/_/-"
[[ "$attempt" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,59}$ ]] || fail "invalid attempt ID"
if ((no_upload && !smoke_test)); then fail "--no-upload is restricted to an explicit --smoke-test; normal B results must be returned"; fi
command -v git >/dev/null || fail "Git is unavailable"
command -v timeout >/dev/null || fail "GNU timeout is required for bounded installation/authorization checks"
command -v "$python_command" >/dev/null || fail "selected Python is unavailable"
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_directory=$(cd -- "$script_directory/.." && pwd -P)
[[ -d "$repo_directory/.git" || -f "$repo_directory/.git" ]] || fail "run from an actual Research4 checkout"
source_branch='research/parallel-worker-b-20261106'
result_branch="${source_branch}-${run_id}"
current_branch=$(git -C "$repo_directory" branch --show-current)
local_a_test=0
if [[ "$current_branch" == 'research/pyscf-spinosin-mechanism-20261031' && "$smoke_test" == 1 && "$no_upload" == 1 ]]; then
  local_a_test=1
else
  [[ "$current_branch" == "$source_branch" || "$current_branch" == "$result_branch" ]] || fail "checkout the B release branch or this run's result branch; main/other runs are forbidden"
fi
PYTHONPATH= "$python_command" - "$repo_directory" "$run_id" "$local_a_test" <<'PY'
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1]);run=sys.argv[2];local_test=sys.argv[3]=='1'
changed=[v.decode() for v in subprocess.check_output(['git','-C',str(root),'diff','--name-only','-z']).split(b'\0') if v]
prefix=f'reports/parallel-cloud-20261106/worker-b/{run}/'
if (local_test and changed) or any(not p.startswith(prefix) for p in changed):
    raise SystemExit('Tracked source/other-run changes must be preserved separately; only this B run\'s saved results may remain modified during recovery.')
PY
git -C "$repo_directory" diff --cached --quiet || fail "preexisting staged edits must be preserved separately"
requirements_file="$repo_directory/requirements/research4-worker-b-20261106.txt"
supervisor="$repo_directory/scripts/research4_parallel_b_supervisor_20261106.py"
[[ -f "$requirements_file" && -f "$supervisor" ]] || fail "B release files are incomplete"

PYTHONPATH= "$python_command" - "$repo_directory" <<'PY'
from pathlib import Path
import importlib.util,sys
root=Path(sys.argv[1]);source=root/'scripts/research4_parallel_b_git_return_20261106.py'
if not source.is_file():raise SystemExit('B Git-return helper is missing; release is incomplete.')
spec=importlib.util.spec_from_file_location('b_origin_check',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.validate_origin(root)
PY

"$python_command" - "$repo_directory" "$venv_relative" <<'PY'
from pathlib import Path
import platform,sys
root=Path(sys.argv[1]).resolve(); relative=Path(sys.argv[2])
if sys.version_info[:2] not in ((3,11),(3,12)):
    raise SystemExit('Python3.11/3.12 required. Existing A calculations used3.12.14; this B interpreter is not yet qualified.')
if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
    raise SystemExit('This release is scoped to Linux x86_64 CPU wheels; platform not qualified.')
if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0]!='.runtime':
    raise SystemExit('A dedicated relative .runtime environment is required; user environments are never overwritten.')
if any((root/p).is_symlink() for p in (relative,*relative.parents)):
    raise SystemExit('Symlink environment path rejected.')
print('Interpreter check:',platform.python_version(),platform.system(),platform.machine(),'; B installation/water qualification still required.')
PY

prepare_flags=(--prepare --run-id "$run_id" --python-executable "$venv_relative/bin/python")
((smoke_test)) && prepare_flags+=(--smoke-test)
((no_upload)) && prepare_flags+=(--no-upload)
if ((check_only)); then
  PYTHONPATH= "$python_command" "$supervisor" "${prepare_flags[@]}" --check-only
  printf '%s\n' "CHECK_ONLY: no packages installed, no branch changed, no quantum calculation or Git push performed."
  exit 0
fi

# Atomically claim the one ROOT-released remote allocation before installing
# or spending compute. The helper performs this account's dry-run and actual
# claim-only non-force push; another host cannot claim the same result branch.
if ((!no_upload)); then
  git -C "$repo_directory" var GIT_AUTHOR_IDENT >/dev/null 2>&1 || fail "Git author identity is not configured in this account; configure it before computing."
  timeout 360s env GIT_TERMINAL_PROMPT=0 PYTHONPATH= "$python_command" "$repo_directory/scripts/research4_parallel_b_git_return_20261106.py" --repo "$repo_directory" --claim-run-id "$run_id" --release "$repo_directory/data/research/parallel_cloud_20261106/worker_b/release_ROOT_APPROVED.json" --execute
fi

venv_directory="$repo_directory/$venv_relative"
marker="$venv_directory/.research4-worker-b-environment.json"
"$python_command" - "$requirements_file" "$venv_directory" "$marker" <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
req,venv,marker=map(Path,sys.argv[1:]);digest=hashlib.sha256(req.read_bytes()).hexdigest()
if venv.exists():
    if not marker.is_file():raise SystemExit('Existing environment lacks B ownership marker; refusing to overwrite it. Choose a NEW --venv path.')
    m=json.loads(marker.read_text())
    if m.get('owner')!='research4-worker-b-20261106' or m.get('requirements_sha256')!=digest:
        raise SystemExit('Environment ownership/dependency pin mismatch; choose a NEW --venv path.')
else:
    venv.mkdir(parents=True)
    marker.write_text(json.dumps({'owner':'research4-worker-b-20261106','requirements_sha256':digest,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'INITIALIZING_INSTALL_NOT_YET_VERIFIED'},indent=2)+'\n')
PY
if [[ ! -x "$venv_directory/bin/python" ]]; then
  "$python_command" -m venv "$venv_directory"
fi
timeout 1800s env PYTHONPATH= "$venv_directory/bin/python" -m pip install --no-input --disable-pip-version-check --timeout 30 --retries 2 -r "$requirements_file"
PYTHONPATH= "$venv_directory/bin/python" - "$marker" <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import importlib.metadata,json,platform,sys
expected={'numpy':'2.3.5','scipy':'1.17.0','pyscf':'2.14.0','geometric':'1.1.1','rdkit':'2026.9.1'}
actual={k:importlib.metadata.version(k) for k in expected}
if actual!=expected:raise SystemExit('Actual installed versions do not match the released CPU requirements.')
p=Path(sys.argv[1]);m=json.loads(p.read_text());m.update(status='DEPENDENCIES_INSTALLED_WATER_NOT_YET_QUALIFIED',python=platform.python_version(),packages=actual,verified_at_utc=datetime.now(timezone.utc).isoformat());p.write_text(json.dumps(m,indent=2)+'\n')
print('Actual B dependency installation verified. Water scientific qualification has not run yet.')
PY

prepare_receipt=$(PYTHONPATH= "$venv_directory/bin/python" "$supervisor" "${prepare_flags[@]}")
mapfile -t receipt_fields < <(printf '%s' "$prepare_receipt" | PYTHONPATH= "$venv_directory/bin/python" -c 'import json,sys;j=json.load(sys.stdin);print(j.get("manifest_path",j.get("manifest","")));print(j["manifest_sha256"]);print(j.get("authorization_path",j.get("authorization","")));print(j["authorization_sha256"])')
[[ ${#receipt_fields[@]} == 4 ]] || fail "supervisor did not return the four pinned manifest/authorization receipt fields"
[[ -n "${receipt_fields[0]}" && -n "${receipt_fields[2]}" ]] || fail "supervisor manifest/authorization paths are missing"
launch_flags=(--manifest "${receipt_fields[0]}" --manifest-sha256 "${receipt_fields[1]}" --authorization "${receipt_fields[2]}" --authorization-sha256 "${receipt_fields[3]}" --attempt "$attempt" --execute)
((detach)) && launch_flags+=(--detach)
if ((local_a_test)); then
  printf '%s\n' 'NO_REMOTE_B_RUN: local water-only/no-upload smoke; this is not evidence of another account or host execution.'
fi
cd -- "$repo_directory"
PYTHONPATH= "$venv_directory/bin/python" "$supervisor" "${launch_flags[@]}"
