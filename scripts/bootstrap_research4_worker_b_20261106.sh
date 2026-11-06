#!/usr/bin/env bash
# Standalone bootstrap. Never runs host-A science, resets main, or imports A credentials.
set -euo pipefail
umask 077

SOURCE_BRANCH='research/parallel-worker-b-20261106'
SOURCE_COMMIT='c2284ad4755b223830aeb2283d7491a3a3f35a28'
RUN_ID='B20261106-01'
ORIGIN='https://github.com/jackzhu119/research4.git'
LAUNCHER_SHA='2c767fdd7a58c55d63fe87dfe992e99859d1337cee532f0c220ff04eb1aaa4c1'
check_only=0; device_login=0
if [[ -d /workspace && -w /workspace ]]; then repo_directory='/workspace/research4-worker-b'; else repo_directory="$PWD/research4-worker-b"; fi

fail() { printf '%s\n' "BLOCKED: $*" >&2; exit 2; }
guidance() { printf '%s\n' 'Reconnect/authorize GitHub in this Codex account and check its GitHub/network access. A public-repository HTTP403 can be a proxy/policy issue; no authentication or network bypass is attempted.' >&2; }
while (($#)); do
  case "$1" in
    --check-only) check_only=1; shift;;
    --device-login) device_login=1; shift;;
    --repo-directory) (($# >= 2)) || fail 'missing repo directory'; repo_directory=$2; shift 2;;
    -h|--help)
      printf '%s\n' 'Usage: bash bootstrap_research4_worker_b_20261106.sh [--check-only] [--device-login] [--repo-directory NEW_PATH]' \
        'Default: bounded tool setup, fixed-release sparse clone, this account Git write check, original B launcher --check-only then --detach.' \
        '--check-only: no clone, installation, network, Git command/write or quantum calculation.' \
        '--device-login: optional OWN web authorization only after write failure and no inherited Git authentication; requires /dev/tty.'
      exit 0;;
    *) fail "unknown argument $1";;
  esac
done

choose_python() {
  local candidate
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys;raise SystemExit(sys.version_info[:2] not in ((3,11),(3,12)))' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"; return 0
    fi
  done
  return 1
}

if ((check_only)); then
  printf '%s\n' 'CHECK_ONLY_NO_SETUP_OR_COMPUTE' "Target: $repo_directory" "Release: $SOURCE_BRANCH @ $SOURCE_COMMIT" "Fixed run: $RUN_ID"
  for tool in git timeout apt-get gh; do
    if command -v "$tool" >/dev/null; then printf '%s\n' "Tool $tool: available"; else printf '%s\n' "Tool $tool: absent (not installed in check-only)"; fi
  done
  if candidate=$(choose_python); then printf '%s\n' "Compatible Python: $candidate"; else printf '%s\n' 'Python3.11/3.12: unavailable (not installed in check-only)'; fi
  if [[ -f "$repo_directory/scripts/run_research4_worker_b_20261106.sh" ]]; then
    bash -n "$repo_directory/scripts/run_research4_worker_b_20261106.sh"
    if command -v sha256sum >/dev/null; then
      actual=$(sha256sum "$repo_directory/scripts/run_research4_worker_b_20261106.sh"); actual=${actual%% *}
      [[ "$actual" == "$LAUNCHER_SHA" ]] || fail 'existing launcher differs from the frozen release'
      printf '%s\n' 'Existing launcher bytes/syntax match the frozen release; Git identity was not queried.'
    fi
  else
    printf '%s\n' 'Frozen checkout is absent/incomplete; no clone attempted.'
  fi
  exit 0
fi

command -v timeout >/dev/null || fail 'GNU timeout is required for bounded setup; choose a Linux cloud image with coreutils.'
log_directory=$(mktemp -d "${TMPDIR:-/tmp}/research4-b-bootstrap.XXXXXX")
setup_log="$log_directory/setup.log"
apt_install() {
  local -a elevated=()
  command -v apt-get >/dev/null || fail 'missing basic tools and apt-get unavailable; use a Python3.11/3.12 Linux cloud image'
  if [[ $(id -u) != 0 ]]; then
    command -v sudo >/dev/null && sudo -n true >/dev/null 2>&1 || fail 'missing basic tools; noninteractive root/sudo is unavailable'
    elevated=(sudo -n -E)
  fi
  timeout 300s "${elevated[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update </dev/null >"$setup_log" 2>&1 || { guidance; fail 'bounded apt update failed; no proxy/CA/source configuration changed'; }
  timeout 600s "${elevated[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@" </dev/null >>"$setup_log" 2>&1 || fail 'bounded basic-tool installation failed; protected local setup log retained'
}
packages=()
command -v git >/dev/null || packages+=(git ca-certificates)
if ! python_command=$(choose_python); then packages+=(python3 python3-venv); fi
if ((${#packages[@]})); then apt_install "${packages[@]}"; fi
python_command=$(choose_python) || fail 'Python3.11/3.12 unavailable after basic setup; newer/older Python is not qualified. Use a compatible cloud image.'
if ! "$python_command" -c 'import venv,ensurepip' >/dev/null 2>&1; then
  minor=$("$python_command" -c 'import sys;print("%s.%s"%sys.version_info[:2])')
  apt_install "python${minor}-venv"
fi
command -v git >/dev/null || fail 'Git remains unavailable'

if [[ ! -e "$repo_directory" ]]; then
  timeout 180s env GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse --single-branch --branch "$SOURCE_BRANCH" "$ORIGIN" "$repo_directory" >"$setup_log" 2>&1 || { guidance; fail 'fixed public release clone failed; partial files preserved, no computation started'; }
elif [[ ! -d "$repo_directory/.git" && ! -f "$repo_directory/.git" ]]; then
  fail 'target exists and is not a known Git checkout; choose a NEW --repo-directory, existing files are never overwritten'
fi
repo_directory=$(cd -- "$repo_directory" && pwd -P)

# Read the origin internally without displaying possible userinfo/credentials.
"$python_command" - "$repo_directory" "$SOURCE_BRANCH" "$SOURCE_COMMIT" "$RUN_ID" <<'PY'
from pathlib import Path
from urllib.parse import urlsplit
import subprocess,sys
root,branch,pin,run=sys.argv[1:]
def git(*args):return subprocess.check_output(['git','-C',root,*args],stderr=subprocess.DEVNULL,text=True,timeout=30).strip()
url=git('remote','get-url','origin')
if url.startswith('git@github.com:'):canonical='https://github.com/'+url.split(':',1)[1]
else:
    value=urlsplit(url)
    if value.hostname!='github.com' or value.scheme not in ('https','ssh'):raise SystemExit('BLOCKED: origin host/protocol mismatch')
    canonical='https://github.com/'+value.path.lstrip('/')
if canonical.removesuffix('.git')!='https://github.com/jackzhu119/research4':raise SystemExit('BLOCKED: origin repository mismatch')
current=git('branch','--show-current');head=git('rev-parse','HEAD')
if current==branch:
    if head != pin:raise SystemExit('BLOCKED: source branch differs from frozen B commit; no reset or checkout is attempted')
elif current==branch+'-'+run:
    if subprocess.run(['git','-C',root,'merge-base','--is-ancestor',pin,'HEAD'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        raise SystemExit('BLOCKED: result branch is not descended from the frozen release')
else:raise SystemExit('BLOCKED: main/unknown branch; no reset or branch replacement is attempted')
PY
timeout 120s git -C "$repo_directory" sparse-checkout add scripts requirements docs data/research/parallel_cloud_20261106 >"$setup_log" 2>&1 || { guidance; fail 'sparse release materialization failed'; }
launcher="$repo_directory/scripts/run_research4_worker_b_20261106.sh"
actual=$("$python_command" - "$launcher" <<'PY'
import hashlib,sys
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())
PY
)
[[ "$actual" == "$LAUNCHER_SHA" ]] || fail 'launcher bytes differ from frozen B release'
if ! git -C "$repo_directory" config --local --get user.name >/dev/null 2>&1 && ! git -C "$repo_directory" config --get user.name >/dev/null 2>&1; then git -C "$repo_directory" config --local user.name 'Research4 Worker B'; fi
if ! git -C "$repo_directory" config --local --get user.email >/dev/null 2>&1 && ! git -C "$repo_directory" config --get user.email >/dev/null 2>&1; then git -C "$repo_directory" config --local user.email 'research4-worker-b@users.noreply.github.com'; fi

result_branch="${SOURCE_BRANCH}-${RUN_ID}"
if ! timeout 120s env GIT_TERMINAL_PROMPT=0 git -C "$repo_directory" push --dry-run origin "HEAD:refs/heads/$result_branch" >"$setup_log" 2>&1; then
  configured=0
  if git -C "$repo_directory" config --name-only --get-regexp '(^credential\..*helper$|^credential\.helper$|^http\..*extraheader$|^http\.extraheader$)' >"$log_directory/auth-config-names" 2>/dev/null; then configured=1; fi
  for key in GH_TOKEN GITHUB_TOKEN GIT_ASKPASS SSH_ASKPASS GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM GH_CONFIG_DIR; do [[ -v "$key" ]] && configured=1; done
  if ((!device_login || configured)); then guidance; fail 'Git write preflight failed; existing authentication is preserved. Use this account Codex GitHub connection; optional --device-login requires no inherited Git authentication.'; fi
  if ! ( : </dev/tty >/dev/tty ) 2>/dev/null; then fail 'device login needs a real terminal (/dev/tty); reconnect GitHub in Codex or run with tty:true. No PAT is read from stdin.'; fi
  if ! command -v gh >/dev/null; then apt_install gh; fi
  auth_directory="$repo_directory/.runtime/worker-b-device-auth"
  [[ ! -e "$auth_directory" ]] || fail 'isolated device-auth directory already exists; preserve it and use the configured account connection'
  mkdir -p "$auth_directory"
  # All gh state/global-git side effects are confined to this new ignored local
  # directory, not existing account credentials or the user's global Git config.
  if ! timeout 600s env GH_CONFIG_DIR="$auth_directory" GIT_CONFIG_GLOBAL="$auth_directory/isolated_gitconfig" gh auth login --hostname github.com --git-protocol https --web </dev/tty >/dev/tty 2>/dev/tty; then
    guidance; fail 'own web/device authorization failed or timed out; no network bypass, PAT import or computation attempted'
  fi
  printf -v quoted_auth '%q' "$auth_directory"
  git -C "$repo_directory" config --local credential.https://github.com.helper "!env GH_CONFIG_DIR=$quoted_auth gh auth git-credential"
  timeout 120s env GIT_TERMINAL_PROMPT=0 git -C "$repo_directory" push --dry-run origin "HEAD:refs/heads/$result_branch" >"$setup_log" 2>&1 || { guidance; fail 'write access still unavailable after own authorization; no computation started'; }
fi

cd -- "$repo_directory"
bash "$launcher" --run-id "$RUN_ID" --check-only
bash "$launcher" --run-id "$RUN_ID" --detach
printf '%s\n' 'B launcher returned. Installation, claim, actual jobs and results are separate states; inspect its real receipt/state. No new schedule or host-A job was created.'
