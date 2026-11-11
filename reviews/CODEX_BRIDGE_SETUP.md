# ChatGPT → GitHub → local Codex bridge

This is an experimental, owner-approved code-editing bridge. It is separate from the existing GPU production runner.

## Installation (in a new WSL2 Ubuntu terminal)

```bash
gh label create codex-task --repo zach102824/qumode_qsk_gpu --color 6F42C1 --description "Owner-approved local Codex coding task" --force
git clone --branch dev/codex-debug https://github.com/zach102824/qumode_qsk_gpu.git ~/projects/qsk_codex_bridge
cd ~/projects/qsk_codex_bridge
python3 reviews/codex_bridge.py start
```

Leave this terminal open. No GPU production is launched by this worker. If a new branch checkout exists already, do not overwrite it. Stop with Ctrl-C. State and results are under `~/.local/state/qsk-codex-bridge/`.

## Task flow

1. In ChatGPT, ask to change code. ChatGPT creates an issue labeled `codex-task` with a short specific instruction.
2. An unchanged owner comment `/codex-approve <sha256-of-exact-issue-body>` authorizes this issue. (The GitHub connector can post this approval when specifically requested here.)
3. The local worker polls issues, clones a separate task checkout, executes `codex exec` in workspace-write / never-escalate mode, and validates changed paths and Python syntax.
4. If it finds permissible changes, it pushes a **separate task branch** and opens a **draft pull request** against `dev/codex-debug`; otherwise it posts a failure or no-change explanation.
5. The worker publishes a status comment to the issue. ChatGPT can read this when asked, and can review the proposed patch. It cannot read local raw data unless separately and explicitly published.

No unattended merge, no production or GPU task, no change to the active experiment checkout. `protocol/`, `cpu_oracle/`, `GPU_test/`, stamps and other frozen/secret-sensitive paths are blocked from publishing. The bridge itself cannot be automatically modified through the bridge.

While running, the bridge posts **only aggregate file-count progress** to [issue #3](https://github.com/zach102824/qumode_qsk_gpu/issues/3) from the existing checkout at `~/projects/qumode_qsk_gpu`. This repository and issue are **public**: do not put unpublished raw numerical data, full logs, credentials or private files into issue descriptions.

This is a workflow boundary, not a hard security boundary against a malicious local user or compromised Codex installation. Review every draft PR before merging. A prompt may fail if Codex authentication or WSL sandboxing is unavailable; failures are reported rather than bypassing protection. Each numeric issue is claimed at most once locally; create a new issue for a deliberate retry.

The separate checkout is critical because the active production source may have uncommitted changes, and modifying the source mid-run can invalidate the physics provenance gate.
