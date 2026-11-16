"""Serial, resumable research jobs under one persistent CPU-second budget.

CLI: ``python species_background_budget.py run --manifest jobs.json --state-dir DIR``
Replace ``run`` with ``resume``, ``status`` or ``stop`` as needed. Manifest jobs
have id, command (argv list), optional cwd/env/inputs/outputs/max_retries (0 or 1).
Optional postprocess_cmd is an argv list run as the last budgeted job.

CPU seconds include supervisor CPU, reaped-child RUSAGE_CHILDREN and current
job-session process CPU (including threads and waited descendants). Wall time
is reported separately. Linux RLIMIT_CPU is per process and integer-valued;
an aggregate watchdog and termination reserve govern the shared budget. A
writable cgroup would be required for an exact last-tick group CPU cutoff.
Workers should checkpoint on SIGTERM; supervisor checkpoints cannot serialize
an arbitrary model. An unclean supervisor crash fails closed on resume rather
than silently losing unobserved CPU consumption or adopting a reused PID.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = {key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TORCH_NUM_THREADS')}
CLOCK_TICKS = os.sysconf('SC_CLK_TCK')
TOKEN_KEY = 'SPECIES_SUPERVISOR_JOB_TOKEN'


class ProvenanceError(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def proc_stat(pid):
    """/proc/pid/stat aggregates all process threads; do not sum task/* again."""
    try:
        text = Path(f'/proc/{int(pid)}/stat').read_text()
        values = text[text.rfind(')') + 2:].split()
        return {'pid': int(pid), 'state': values[0], 'ppid': int(values[1]),
                'pgrp': int(values[2]), 'session': int(values[3]),
                'cpu_seconds': sum(int(values[i]) for i in (11, 12, 13, 14)) / CLOCK_TICKS,
                'start_ticks': int(values[19]), 'threads': int(values[17])}
    except (OSError, ValueError, IndexError):
        return None


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def same_process(identity):
    now = proc_stat(identity['pid'])
    return (now is not None and identity.get('boot_id') == boot_id()
            and now['start_ticks'] == identity.get('start_ticks')
            and now['session'] == identity.get('session')
            and now['pgrp'] == identity.get('pgrp'))


def has_token(pid, token):
    try:
        # Never print or retain the remaining process environment.
        return (TOKEN_KEY + '=' + token).encode() in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    except OSError:
        return False


def owned_members(identity):
    if identity.get('boot_id') != boot_id():
        return []
    processes = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            item = proc_stat(int(path.name))
            if item:
                processes[item['pid']] = item
    root_matches = same_process(identity)
    known = {int(row['pid']): row for row in identity.get('known_descendants', [])}
    selected = set()
    for pid, item in processes.items():
        previous = known.get(pid)
        matches_known = previous and previous['start_ticks'] == item['start_ticks']
        # A descendant may call setsid()/setpgid(). Its lifetime and inherited
        # random token still anchor ownership even after reparenting.
        if (root_matches and item['session'] == identity['session']) or matches_known:
            selected.add(pid)
        elif has_token(pid, identity['token']):
            selected.add(pid)
    if root_matches:
        selected.add(identity['pid'])
    changed = True
    while changed:
        additions = {pid for pid, item in processes.items() if item['ppid'] in selected} - selected
        changed = bool(additions)
        selected.update(additions)
    members = [processes[pid] for pid in selected]
    identity['known_descendants'] = [
        {key: member[key] for key in ('pid', 'start_ticks', 'session', 'pgrp')}
        for member in members]
    return members


def signal_owned_groups(identity, number):
    members = owned_members(identity)
    for group in {m['pgrp'] for m in members if m['state'] != 'Z'}:
        # The supervisor itself and unrelated sessions are never candidates.
        if group == os.getpgrp() or group <= 1:
            raise RuntimeError('Refusing to signal supervisor or system process group')
        try:
            os.killpg(group, number)
        except ProcessLookupError:
            pass
    return bool(members)


@contextmanager
def exclusive_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'supervisor.lock'
    with path.open('a+') as handle:
        os.chmod(path, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another supervisor holds this run lock') from exc
        yield


def cpu_usage(kind):
    usage = resource.getrusage(kind)
    return usage.ru_utime + usage.ru_stime


class BudgetSupervisor:
    def __init__(self, state_dir, *, budget_cpu_seconds=90000., poll_seconds=.5,
                 heartbeat_seconds=5., termination_grace_seconds=2., cpu_reserve_seconds=None):
        if budget_cpu_seconds <= 0 or poll_seconds <= 0 or heartbeat_seconds <= 0 or termination_grace_seconds < 0:
            raise ValueError('Positive budget/poll/heartbeat and nonnegative termination grace required')
        self.directory = Path(state_dir).resolve()
        self.budget = float(budget_cpu_seconds)
        self.poll = float(poll_seconds)
        self.heartbeat = float(heartbeat_seconds)
        self.grace = float(termination_grace_seconds)
        self.reserve = float(cpu_reserve_seconds if cpu_reserve_seconds is not None
                             else min(self.budget * .1, max(5., self.grace + 2 * self.poll)))
        if self.reserve < 0 or self.reserve >= self.budget:
            raise ValueError('CPU reserve must be >=0 and smaller than total budget')
        self.state_path = self.directory / 'state.json'
        self.stop_reason = None
        self.active = None
        self.last_write = 0.

    def _total_cpu(self):
        active_cpu = sum(m['cpu_seconds'] for m in owned_members(self.active)) if self.active else 0.
        total = self.base_cpu + max(0., cpu_usage(resource.RUSAGE_SELF) - self.self_start)
        total += max(0., cpu_usage(resource.RUSAGE_CHILDREN) - self.children_start) + active_cpu
        # Sampling/reaping races can cause transient decreases; never refund CPU.
        self.maximum_cpu = max(self.maximum_cpu, total)
        return self.maximum_cpu

    def _save(self, force=False):
        if not force and time.monotonic() - self.last_write < self.heartbeat:
            return
        self.state['spent_cpu_seconds'] = self._total_cpu()
        self.state['remaining_cpu_seconds'] = max(0., self.budget - self.state['spent_cpu_seconds'])
        self.state['active'] = self.active
        self.state['heartbeat_unix'] = time.time()
        self.state['wall_seconds_this_invocation'] = time.monotonic() - self.wall_start
        atomic_json(self.state_path, self.state)
        self.last_write = time.monotonic()

    def _request_reason(self):
        path = self.directory / 'stop_requested.json'
        if path.is_file():
            request = json.loads(path.read_text())
            if request.get('run_id') == self.state['run_id']:
                self.stop_reason = 'user_stop'
        return self.stop_reason

    def _input_hashes(self, job, cache=None):
        cache = {} if cache is None else cache
        hashes = {}
        for entry in job.get('inputs', []):
            path = (Path(job['cwd']) / entry).resolve()
            if not path.is_file():
                raise ProvenanceError(f'Missing declared input: {path}')
            if str(path) not in cache:
                cache[str(path)] = digest(path)
            hashes[str(path)] = cache[str(path)]
        return hashes

    def _verify_frozen(self, jobs):
        if digest(__file__) != self.state['supervisor_code_sha256']:
            raise ProvenanceError('Supervisor source code changed during this run')
        cache = {}
        for job in jobs:
            if self._input_hashes(job, cache) != self.state['input_hashes'][job['id']]:
                raise ProvenanceError('Declared job input changed: ' + job['id'])
        if self.manifest.get('protocol_path'):
            path = Path(self.manifest['protocol_path']).resolve()
            if digest(path) != self.manifest.get('protocol_sha256'):
                raise ProvenanceError('Frozen protocol digest differs')

    def _outputs(self, job):
        output = {}
        for entry in job.get('outputs', []):
            path = (Path(job['cwd']) / entry).resolve()
            if not path.is_file():
                raise FileNotFoundError('Missing expected output: ' + str(path))
            output[str(path)] = digest(path)
        return output

    def _can_skip(self, job):
        previous = self.state['jobs'].get(job['id'], {})
        if previous.get('status') != 'completed':
            return False
        receipt_path = self.directory / 'receipts' / (job['id'] + '.json')
        try:
            receipt = json.loads(receipt_path.read_text())
            return (digest(receipt_path) == previous['receipt_sha256']
                    and receipt['job_sha256'] == object_hash(job)
                    and receipt['input_hashes'] == self._input_hashes(job)
                    and receipt['output_hashes'] == self._outputs(job)
                    and digest(receipt['log_path']) == receipt['log_sha256'])
        except (OSError, KeyError, ValueError):
            return False

    def _terminate(self, process, reason):
        self.state['termination_reason'] = reason
        signal_owned_groups(self.active, signal.SIGTERM)
        until = time.monotonic() + self.grace
        while time.monotonic() < until:
            process.poll()
            self._save(force=True)
            if not any(m['state'] != 'Z' for m in owned_members(self.active)):
                break
            if self._total_cpu() >= self.budget - min(.02, self.reserve / 2):
                break
            time.sleep(min(self.poll, .05))
        signal_owned_groups(self.active, signal.SIGKILL)
        try:
            process.wait(timeout=3.)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Owned process failed to exit after SIGKILL')
        self._save(force=True)

    def _run_job(self, job):
        previous = self.state['jobs'].get(job['id'], {})
        attempt = int(previous.get('attempts', 0)) + 1
        path = self.directory / 'logs' / (job['id'] + f'.attempt{attempt}.log')
        path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        environment = os.environ.copy()
        environment.update({str(k): str(v) for k, v in job.get('env', {}).items()})
        environment.update(THREAD_ENV)
        environment[TOKEN_KEY] = token
        remaining = self.budget - self._total_cpu()

        def limits():
            os.umask(0o077)
            # All children inherit the remaining global ceiling; it is never
            # reset to 90000 for later jobs. Aggregate descendants are watched.
            hard = int(math.floor(remaining - min(self.reserve, remaining / 2)))
            if hard >= 1:
                resource.setrlimit(resource.RLIMIT_CPU, (max(1, hard - 1), hard))

        record = {'status': 'running', 'attempts': attempt, 'job_sha256': object_hash(job),
                  'started_unix': time.time(), 'log_path': str(path)}
        self.state['jobs'][job['id']] = record
        process = None
        try:
            with path.open('xb') as log:
                process = subprocess.Popen(job['command'], cwd=job['cwd'], env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    preexec_fn=limits, close_fds=True)
                identity = proc_stat(process.pid)
                if identity is None:
                    process.wait()
                    raise RuntimeError('Child exited before its identity could be recorded')
                self.active = {key: identity[key] for key in ('pid', 'pgrp', 'session', 'start_ticks')}
                self.active.update({'boot_id': boot_id(), 'token': token, 'job_id': job['id'],
                                    'remaining_cpu_at_spawn': remaining})
                self._save(force=True)
                interrupted = None
                while process.poll() is None:
                    if self._request_reason():
                        interrupted = self.stop_reason
                    elif self._total_cpu() >= self.budget - self.reserve:
                        interrupted = 'cpu_budget_exhausted'
                    if interrupted:
                        self._terminate(process, interrupted)
                        break
                    self._save()
                    time.sleep(self.poll)
                code = process.wait()
                if any(m['state'] != 'Z' for m in owned_members(self.active)):
                    self._terminate(process, 'orphan_descendants_after_worker_exit')
                    code = code or -1
                self._save(force=True)
                self.active = None
                record.update({'exit_code': code, 'finished_unix': time.time()})
                if interrupted:
                    record.update({'status': 'interrupted', 'reason': interrupted})
                    self.state['status'] = 'budget_exhausted' if interrupted == 'cpu_budget_exhausted' else 'paused'
                    return False
            if code != 0:
                raise RuntimeError('Worker returned nonzero exit code: ' + str(code))
            self._verify_frozen(self.jobs)
            receipt = {'job_sha256': object_hash(job), 'input_hashes': self._input_hashes(job),
                       'output_hashes': self._outputs(job), 'log_path': str(path),
                       'log_sha256': digest(path), 'exit_code': 0, 'finished_unix': time.time()}
            receipt_path = self.directory / 'receipts' / (job['id'] + '.json')
            atomic_json(receipt_path, receipt)
            record.update({'status': 'completed', 'receipt_sha256': digest(receipt_path)})
        except ProvenanceError:
            record.update({'status': 'provenance_error'})
            raise
        except Exception as error:
            if process is not None and self.active is not None:
                self._terminate(process, 'worker_supervision_error')
                self.active = None
            record.update({'status': 'failed', 'error': type(error).__name__ + ': ' + str(error)})
        finally:
            self._save(force=True)
        return True

    def run(self, manifest, *, resume=False):
        manifest = json.loads(json.dumps(manifest))
        authorized_cap = float(manifest.get('cpu_budget_seconds', manifest.get('budget_cpu_seconds', 90000.)))
        if not math.isfinite(authorized_cap) or authorized_cap <= 0 or self.budget > authorized_cap:
            raise ValueError('Requested CPU budget exceeds the positive manifest-authorized ceiling')
        self.manifest = manifest
        self.stop_reason = None
        jobs = list(manifest.get('jobs', []))
        if manifest.get('postprocess_cmd'):
            jobs.append({'id': '__postprocess__', 'command': manifest['postprocess_cmd'],
                         'cwd': manifest.get('cwd', str(ROOT)), 'inputs': manifest.get('postprocess_inputs', []),
                         'outputs': manifest.get('postprocess_outputs', []), 'max_retries': 0})
        seen = set()
        for job in jobs:
            if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', str(job.get('id', ''))) or job['id'] in seen:
                raise ValueError('Unique safe job IDs required')
            seen.add(job['id'])
            if not isinstance(job.get('command'), list) or not job['command'] or not all(isinstance(v, str) for v in job['command']):
                raise ValueError('command must be a nonempty argv list; shell text is not accepted')
            if job.get('max_retries', 0) not in (0, 1):
                raise ValueError('At most one explicitly declared retry is supported')
            job['cwd'] = str(Path(job.get('cwd', manifest.get('cwd', str(ROOT)))).resolve())
        self.jobs = jobs
        frozen_manifest = {'manifest': manifest, 'normalized_jobs': jobs, 'budget_cpu_seconds': self.budget}
        with exclusive_lock(self.directory):
            self.wall_start = time.monotonic()
            self.self_start, self.children_start = cpu_usage(resource.RUSAGE_SELF), cpu_usage(resource.RUSAGE_CHILDREN)
            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text())
                if not resume:
                    raise RuntimeError('State already exists; use resume to preserve cumulative CPU accounting')
                if self.state['manifest_sha256'] != object_hash(frozen_manifest):
                    raise ProvenanceError('Manifest or shared CPU budget differs from its frozen version')
                if self.state.get('active'):
                    # An abrupt supervisor death can leave unobserved child CPU.
                    # Do not refund it or signal a PID whose lifetime is unknown.
                    raise RuntimeError('Unclean prior active job: resume blocked; inspect verified identities with status')
                self.base_cpu = float(self.state['spent_cpu_seconds'])
            else:
                self.base_cpu = 0.
                input_cache = {}
                all_input_hashes = {job['id']: self._input_hashes(job, input_cache) for job in jobs}
                self.state = {'schema_version': 1, 'run_id': uuid.uuid4().hex,
                    'manifest_sha256': object_hash(frozen_manifest), 'budget_cpu_seconds': self.budget,
                    'supervisor_code_sha256': digest(__file__), 'spent_cpu_seconds': 0., 'jobs': {},
                    'input_hashes': all_input_hashes,
                    'created_unix': time.time(), 'cpu_budget_unit': 'aggregate process CPU core-seconds, not wall seconds',
                    'enforcement': 'Per-child remaining-budget RLIMIT_CPU plus session aggregate watchdog; termination reserve retained',
                    'cpu_termination_reserve_seconds': self.reserve, 'poll_seconds': self.poll,
                    'heartbeat_seconds': self.heartbeat, 'worker_threads': 1}
                atomic_json(self.directory / 'manifest.snapshot.json', frozen_manifest)
            self.maximum_cpu = self.base_cpu
            self.state['owner'] = {**{k: proc_stat(os.getpid())[k] for k in ('pid', 'start_ticks', 'pgrp', 'session')}, 'boot_id': boot_id()}
            self.state['status'] = 'running'
            stop_path = self.directory / 'stop_requested.json'
            if stop_path.exists():
                stop_path.unlink()
            old_handlers = {}
            for number in (signal.SIGTERM, signal.SIGINT):
                old_handlers[number] = signal.getsignal(number)
                signal.signal(number, lambda signum, frame: setattr(self, 'stop_reason', 'signal_' + signal.Signals(signum).name))
            try:
                self._verify_frozen(jobs)
                self._save(force=True)
                for job in jobs:
                    self._verify_frozen(jobs)
                    if self._can_skip(job):
                        self.state['jobs'][job['id']]['last_resume_verified_skip'] = time.time()
                        continue
                    if self._request_reason():
                        self.state['status'] = 'paused'
                        break
                    if self._total_cpu() >= self.budget - self.reserve:
                        self.state['status'] = 'budget_exhausted'
                        break
                    previous = self.state['jobs'].get(job['id'], {})
                    if previous.get('status') == 'failed' and previous.get('attempts', 0) >= 1 + job.get('max_retries', 0):
                        continue
                    while True:
                        if not self._run_job(job):
                            break
                        current = self.state['jobs'][job['id']]
                        if current['status'] != 'failed' or current['attempts'] >= 1 + job.get('max_retries', 0):
                            break
                        if self._total_cpu() >= self.budget - self.reserve or self._request_reason():
                            break
                    if self.state['status'] != 'running':
                        break
                if self.state['status'] == 'running':
                    self.state['status'] = ('completed_with_errors' if any(v['status'] == 'failed' for v in self.state['jobs'].values()) else 'completed')
            except ProvenanceError as error:
                self.state.update({'status': 'blocked_provenance', 'error': str(error)})
            finally:
                self._save(force=True)
                for number, handler in old_handlers.items():
                    signal.signal(number, handler)
            return self.state


def request_stop(state_dir):
    directory = Path(state_dir)
    state = json.loads((directory / 'state.json').read_text())
    atomic_json(directory / 'stop_requested.json', {'run_id': state['run_id'], 'requested_unix': time.time()})
    return {'stop_requested': True, 'run_id': state['run_id'],
            'supervisor_identity_matches': same_process(state['owner'])}


def status(state_dir):
    state = json.loads((Path(state_dir) / 'state.json').read_text())
    state['supervisor_identity_matches'] = same_process(state['owner'])
    state['active_identity_matches'] = same_process(state['active']) if state.get('active') else False
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'resume', 'status', 'stop'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--budget-cpu-seconds', type=float)
    args = parser.parse_args()
    if args.action == 'status':
        result = status(args.state_dir)
    elif args.action == 'stop':
        result = request_stop(args.state_dir)
    else:
        if args.manifest is None:
            parser.error('--manifest is required for run/resume')
        manifest = json.loads(args.manifest.read_text())
        budget = args.budget_cpu_seconds if args.budget_cpu_seconds is not None else manifest.get('cpu_budget_seconds', manifest.get('budget_cpu_seconds', 90000.))
        supervisor = BudgetSupervisor(args.state_dir, budget_cpu_seconds=budget,
            poll_seconds=manifest.get('poll_seconds', .5), heartbeat_seconds=manifest.get('heartbeat_seconds', 5.),
            termination_grace_seconds=manifest.get('termination_grace_seconds', 2.))
        result = supervisor.run(manifest, resume=args.action == 'resume')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get('status') in ('completed', None) else 1


if __name__ == '__main__':
    raise SystemExit(main())
