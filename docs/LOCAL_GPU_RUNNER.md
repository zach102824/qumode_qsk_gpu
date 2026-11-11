# Local owner-approved GPU queue

This runner stays on `dev/codex-debug`, uses `.venv/bin/python` 3.13.5 in
WSL2 Ubuntu, and executes fixed tasks on CUDA device **1**, which must identify
as RTX 5090. It never fetches branches or executes issue text as code. Keep
the local checkout, environment, and runner state private to the trusted OS user.
This is a command allowlist with isolated output paths, not a sandbox for malicious
local Python source. Review local code before approving its fingerprint.

## Commands

From the repository root, set `GPU_RUNNER_TOKEN` in your terminal environment
using a fine-grained GitHub token restricted to `zach102824/qumode_qsk_gpu`,
with Metadata read and Issues read/write. No Contents write permission is needed.
Do not put the token in issues, command arguments, tracked files, or logs.
The implementation uses the [GitHub issues API](https://docs.github.com/en/rest/issues/issues)
and [issue comments API](https://docs.github.com/en/rest/issues/comments).

Start in a dedicated WSL terminal (foreground; keeps errors visible):

```bash
.venv/bin/python scripts/local_gpu_runner.py start
```

Stop from a second terminal:

```bash
.venv/bin/python scripts/local_gpu_runner.py stop
```

Ctrl-C or SIGTERM also stops it. A file watcher notices stop requests within
one second, including while a child runs. HTTP requests have a 30-second timeout;
a request already in flight can take that long to return. The runner sends
SIGTERM then SIGKILL to the child process group, allowing up to five seconds
for graceful shutdown. A GPU driver hang can require manual intervention.
A new start clears the stop file only after acquiring the exclusive local lock.

One polling pass, executing at most one approved task:

```bash
.venv/bin/python scripts/local_gpu_runner.py once
```

Defaults: poll every 60 seconds, each task timeout 7200 seconds.
Local `--interval` and `--timeout` flags can adjust these bounds; issue content
cannot. Only one runner may hold the local flock. Do not start another runner
from a second checkout or machine: coordination is local, not distributed.

## Request and owner approval

Create an ordinary open issue labeled `gpu-run`. Its entire body must be
one of these exact task names (surrounding whitespace is accepted, but hashed):

| Issue body | Fixed behavior |
| --- | --- |
| `gpu-validation` | RTX 5090 check, full GPU validation on device 1, unchanged default tolerances and library pins; writes a new stamp only inside its task snapshot |
| `production-dry-run` | `run_production.py --dry-run --round 1`; planning only, no CUDA acquisition |
| `production-gate-verification` | Acquire device 1, check RTX 5090, assert and reverify existing gate; never dispatch jobs |

Approval must be a new, unedited comment by the **personal repository owner**.
Collaborators, issue authors, labels, bots, and author-association strings do not
authorize anything. The runner obtains the owner's numeric GitHub user ID from
the fixed repository API and compares it to the comment author's numeric ID.
Organization-owned repositories are refused in version 1.

Compute the local code fingerprint:

```bash
.venv/bin/python scripts/local_gpu_runner.py fingerprint
```

Compute SHA-256 of the issue's exact raw UTF-8 body, including any newline.
For a body consisting of just `production-dry-run`, without a trailing newline:

```bash
.venv/bin/python -c 'import hashlib; print(hashlib.sha256(b"production-dry-run").hexdigest())'
```

The owner then posts this single line, replacing both placeholders:

```text
/gpu-approve BODY_SHA256 LOCAL_CODE_SHA256
```

The latest owner comment starting with `/gpu-approve` or `/gpu-revoke` wins.
Posting `/gpu-revoke` revokes a pending approval. Editing an approval invalidates
it; post a new comment instead. Any body or local code change invalidates the
approval. Labels and issue state are rechecked using a fresh issue read before
claiming. Code changes while the daemon runs block new jobs until it is restarted.
Approvals apply to the reviewed local snapshot, including uncommitted changes,
not to remote HEAD. Nothing is pulled from GitHub. Reviewers can inspect the
retained snapshot and its fingerprint after execution.

There is an unavoidable small race between the final GitHub read and local
launch: revocation after that read does not cancel an already claimed task.
Use the local stop command to terminate active work. Only approve trusted tasks.

## Artifacts, failures and recovery

`.local-gpu-runner/` is ignored by Git. Each GitHub numeric issue ID gets a
new directory with `issue.json`, `command.json`, `complete.log`, `result.json`
and `workspace/`. The workspace retains all numerical reports, scratch outputs,
and validation stamps. Each snapshot copies only source, scripts, docs, frozen
fixtures/protocol, oracle, and the existing PASS stamp. Existing research outputs
are never copied or touched. Bytecode and symlinks are excluded/rejected.
The original `GPU_TEST_PASS.json` is never replaced. Adding runner scripts/docs
changes the gate's source binding, so the preserved old stamp may correctly fail
verification. A new validation stamp lives only in its isolated snapshot; this
runner never promotes it to the original checkout.

SQLite with FULL synchronous commits records a task claim **before** execution.
The issue's immutable numeric ID is the task ID. A claimed, completed, failed,
interrupted or timed-out issue can never execute again with this ledger, even if
closed/reopened, edited, or relabeled. Request a new issue and owner approval
for a deliberate retry. Never delete or reset the ledger to retry work.
After a crash a claim becomes `interrupted`; this may mean the job never
started or that it ran. Inspect artifacts manually. Do not restart until you
have checked for orphan child processes after SIGKILL, host crash, or power loss.
Graceful shutdown and timeouts clean up the whole task process group.

The runner posts a concise claimed status and a final PASS/FAIL, bounded log
excerpt (including traceback when present), exit code, and suggested diagnosis
on failure. Full tracebacks and numerical results stay local. Failures with no
Python exception (e.g. signal, numerical FAIL or timeout) may have no traceback.
Long tracebacks may be truncated on GitHub; inspect the complete local log and
`validation-report.json`. No source edits or tolerance adjustments follow failures.
Final summaries are a durable outbox: publication failures retry without rerunning
the job. A marker check reduces duplicate comments after uncertain API responses;
GitHub comments have no transactional exactly-once guarantee. Failure to publish
the initial claimed status prevents execution and consumes the claim safely.

Network/auth/API failures fail closed for new jobs. API requests use fixed HTTPS
GitHub endpoints, reject redirects, paginate fully, and time out. Tokens are not
passed to child processes. No shell is used, and no arguments, paths, Python code,
URLs or environment overrides are accepted from issues. Task logs may contain
local paths and scientific diagnostics; review this disclosure before approval.

## Safe one-task end-to-end test

No token, network, GPU validation or production execution is needed:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_local_gpu_runner.py' -v
```

The end-to-end test uses an in-memory GitHub API double and a separate persistent
self-test ledger. It executes the **real dry-run planner** in a fresh snapshot,
checks the final publication, polls again to prove it does not run twice, and
hashes the original stamp and research artifacts before/after. Evidence remains
under `.local-gpu-runner/self-test/`. Other tests cover injection, forged/edited
approvals, revocation, PR exclusion, code mismatch, symlink rejection, restart
claims, publication retry, token stripping, timeout and stop behavior.

For a later live smoke test, create a `production-dry-run` issue, approve its
body and current fingerprint as above, then run `once` twice. Inspect the issue
and local ledger: one claim and one execution. Do not approve `gpu-validation`
unless you intend the full fixture validation workload. Production execution
(`--run`), benchmarks, merges, pushes, deletion, frozen physics changes and
relaxed tolerances have no task entry point.

## Implementation safety review

Reviewed the issue-to-launch path and recovery behavior for version 1:

- **Command injection:** issue body selects an exact enum; fixed argv lists use
  `Popen` without a shell. Paths and subprocess environment are local constants.
- **Authorization:** fixed personal repository, numeric owner ID, unedited exact
  approval, raw body hash and local snapshot hash. Closed issues, PRs, missing
  labels, other users and bots fail authorization. Full comment pagination and
  redirect rejection prevent partial checks or credential forwarding. Only
  comments from the authenticated publishing account count as prior summaries.
- **Duplicate execution:** exclusive flock, SQLite primary key, durable claim
  before launch, no failed/interrupted claim retry, independently retried outbox.
  This guarantee requires preserving the local ledger and using one checkout.
- **Timeout/shutdown:** HTTP timeout, bounded subprocess runtime, stop watcher,
  process-group TERM/KILL, and no token in child environment. Abrupt host death
  remains a manual recovery case; it never triggers automatic replay.
- **Production and data:** dry-run argv always includes `--dry-run`, never `--run`;
  gate verification only checks binding. Validation uses a fresh snapshot, so its
  destructive scratch cleanup and stamp writer cannot touch original artifacts.
  No merge, push, checkout, source patch or tolerance override operation exists.
- **Validation compatibility:** the device-name probe does not acquire the package
  runtime before the validator's import-safety check. A regression test protects
  this requirement without initializing CUDA.

Local verification passed 14 offline tests, including a real planner subprocess,
plus the repository static check (43 Python files, 324 start-hash checks, no
import-time CUDA). WSL2 Ubuntu, branch and interpreter preflight passed. The
end-to-end test verified byte-for-byte preservation of existing stamps, research
outputs and validation scratch artifacts. Live GitHub transport and full GPU
numerics were not exercised by this implementation test.
