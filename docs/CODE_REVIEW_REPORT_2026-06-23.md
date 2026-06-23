# Taiji Code Review Report

Date: 2026-06-23
Reviewer: Codex
Scope: strict review of the current working tree, with emphasis on changed files and high-risk runtime paths.

## Review Scope

This pass covered:

- API entrypoints and middleware
- authentication and authorization flow
- workspace, terminal, update, model-switch, and runtime endpoints
- startup/model-loading path
- life scheduler and background-thread behavior
- frontend runtime/auth bootstrap flow
- test suite structure and executability

This is a report-grade review of the current implementation, not a mathematical proof that every line in the repository is defect-free.

## Findings

### P0 - Unauthenticated takeover of auth configuration

Files:

- `api/app.py:63`
- `api/routes_auth.py:55`
- `taiji/services/auth_service.py:59`
- `taiji/core/security.py:412`

Problem:

`/api/auth/enable` is explicitly listed as a public path in the JWT middleware and directly enables authentication with caller-supplied credentials. After auth is already enabled, an unauthenticated caller can still reset the username/password pair.

Evidence:

- `JWTAuthMiddleware.PUBLIC_PATHS` includes `/api/auth/enable`
- `POST /api/auth/enable` calls `auth_service.enable_auth(...)` with no extra gate
- isolated runtime verification showed `POST /api/auth/enable` returned `200` and changed the configured username while other protected endpoints returned `401`

Impact:

Remote attacker can seize administrative control of the application.

Recommendation:

- allow unauthenticated enable only for first-run bootstrap, or remove it from public routes entirely
- require an existing valid token or local-only confirmation when auth is already enabled
- add regression tests for first-run vs post-bootstrap behavior

### P0 - Remote shell exposure through Web terminal

Files:

- `api/routes_terminal.py:54`
- `api/routes_terminal.py:103`
- `api/app.py:70`

Problem:

The Web terminal opens a real shell process (`cmd.exe` on Windows, login shell on Unix). When auth is disabled, `_verify_ws_token()` returns `True`. Separately, `/ws/` is excluded from HTTP JWT middleware by prefix.

Impact:

Default deployment allows browser clients to reach a shell-capable endpoint with no authentication barrier if auth has not been explicitly turned on. This is effectively remote command execution.

Recommendation:

- disable terminal by default
- require explicit setting to enable it
- require token validation regardless of global auth mode, or restrict to loopback/desktop-only sessions
- validate WebSocket origin and add audit logging

### P0 - Update and hot-patch endpoints provide code-execution capability without trust validation

Files:

- `api/routes_update.py:86`
- `api/routes_update.py:107`
- `api/routes_update.py:147`
- `api/routes_update.py:213`
- `api/routes_update.py:269`

Problem:

The update surface can:

- download and install a package from a URL
- upload a zip and install it
- upload arbitrary `.py` patches into `update_code`
- hot-reload modules
- replace the frontend bundle

There is no signature verification, trusted-source restriction, hash pinning, or approval flow beyond generic API access.

Impact:

Any compromised session with API access can transition into persistent arbitrary code execution.

Recommendation:

- treat update/patch operations as privileged admin-only actions
- require signed manifests or pinned hashes
- limit update sources to trusted allowlists
- record immutable audit events for all update operations

### P1 - Public bootstrap contract is broken

Files:

- `api/routes_runtime.py:15`
- `taiji/services/runtime_service.py:80`
- `api/app.py:63`

Problem:

The runtime bootstrap endpoint is documented and implemented as public, but `/api/runtime/bootstrap` is not in `PUBLIC_PATHS`. When auth is enabled, this endpoint returns `401`.

Impact:

The frontend cannot reliably determine whether it should show a login flow or continue loading the runtime shell. This creates startup failures and confusing auth behavior.

Recommendation:

- either make `/api/runtime/bootstrap` truly public as designed
- or change the frontend to use a different explicit unauthenticated probe and update docs accordingly

### P1 - Frontend auth/bootstrap flow does not match backend contract

Files:

- `frontend/src/composables/useApi.js:74`
- `frontend/src/stores/runtimeStore.js:312`
- `frontend/src/composables/useAuth.js:17`
- `taiji/services/runtime_service.py:80`

Problem:

The frontend health loop hits `/api/runtime/status` directly and treats `401` mostly as connectivity degradation. It does not first use the public bootstrap endpoint to determine whether auth is required. `useAuth` says auth state comes from `runtimeStore`, but the runtime store itself depends on the protected status endpoint.

Impact:

When auth is enabled and no token is present, the UI can degrade into a reconnect/error loop instead of a deterministic login-first state.

Recommendation:

- make app startup call `/api/runtime/bootstrap` first
- if `need_login` is true, gate all protected polling behind successful login
- add an integration test covering cold start with auth enabled and no token

### P1 - Workspace root can be repointed to arbitrary directories

Files:

- `api/routes_agent_workspace.py:39`
- `api/routes_agent_workspace.py:104`
- `api/routes_agent_workspace.py:125`
- `api/routes_agent_workspace.py:179`

Problem:

`POST /api/workspace/path` accepts any existing directory. After that, workspace read/write/delete endpoints trust the new directory as the sandbox root.

Impact:

Any caller with workspace API access can rebind the workspace to the user home, project root, or drive root and then read, overwrite, or delete files there through otherwise "safe" workspace APIs.

Recommendation:

- restrict workspace roots to an approved base directory set
- require explicit admin approval for changing workspace root
- log and surface current effective workspace root prominently

### P1 - Static workspace file serving bypasses auth strategy

Files:

- `api/app.py:70`
- `api/app.py:332`

Problem:

`/workspace_data` is mounted as static files and also exempted from JWT middleware through `PUBLIC_PREFIXES`.

Impact:

Files placed under the workspace may be downloadable without the same protections that guard the corresponding API endpoints.

Recommendation:

- remove `/workspace_data` from public prefixes unless anonymous file serving is a deliberate feature
- if public serving is required, separate public artifacts from the editable workspace

### P1 - Test suite is structurally broken under pytest

Files:

- `tests/test_missing_endpoints.py:1`
- `tests/test_missing_endpoints.py:136`
- `tests/auto_test_runner.py:370`
- `tests/test_cuda_engine.py:409`
- `tests/test_model_download.py:239`

Problem:

Several files act like standalone scripts instead of pytest modules. The most serious case is `tests/test_missing_endpoints.py`, which executes network probes at import time and ends with `sys.exit(...)`.

Observed behavior:

- `python -m pytest tests/ -v` collected tests, then failed during collection with `SystemExit: 1`
- the same run also showed endpoint checks depending on an external server at `http://127.0.0.1:8000`

Impact:

CI and local verification cannot be trusted, because collection itself is unstable and environment-dependent.

Recommendation:

- convert script-style tests into real pytest test functions
- remove top-level `sys.exit()` and network side effects from import time
- split integration probes into opt-in suites guarded by markers or fixtures

### P2 - Training config parsing is internally inconsistent

Files:

- `taiji/core/config.py:211`
- `api/main.py:81`

Problem:

`get_config()` attempts to copy a `dataset_path` attribute from parsed args, but `TrainingConfig` does not define it. The training path actually used later is `config.train_file`.

Impact:

CLI configuration can silently fail to propagate the dataset argument, leading to confusing train-mode behavior.

Recommendation:

- align config parsing and runtime usage around one field name
- add a unit test for command-line config propagation

### P2 - Rate-limit design and tests do not validate the real attack surface

Files:

- `api/app.py:40`
- `api/app.py:126`
- `tests/test_missing_endpoints.py:117`

Problem:

High-value endpoints such as settings, system, and workspace are partially exempted or categorized in ways that do not match the tests. The test script expects `/api/health` to hit `429`, while middleware explicitly exempts health in one path and the script runs outside the real in-process test client.

Impact:

The project has rate-limit code, but the current tests do not establish confidence that meaningful abuse scenarios are covered.

Recommendation:

- define rate-limit policy per endpoint family
- test it with FastAPI test clients rather than external polling scripts
- verify protected write endpoints specifically

### P2 - Background-thread lifecycle is fragile

Files:

- `api/routes_model_switch.py:71`
- `api/routes_update.py:95`
- `api/routes_system.py:89`
- `taiji/core/model_loader.py:507`
- `taiji/life/life_scheduler.py:142`

Problem:

The code heavily relies on daemon threads for model switching, updates, restart orchestration, life scheduling, and auto-reload. Several flows have no durable job state, cancellation handshake, or shutdown coordination. `restart_system()` eventually calls `os._exit(0)`.

Impact:

Partial writes, abandoned work, and hard-to-reproduce race conditions are likely under load or during restart/update operations.

Recommendation:

- centralize long-running job state
- avoid `os._exit(0)` except as last-resort crash handling
- add explicit join/cancel semantics where operations mutate config or model state

## Verification Performed

Commands run:

- `git status --short`
- `git diff --stat`
- targeted `rg -n` review across API, runtime, life, frontend, and tests
- isolated auth-path verification using a temporary `TAIJI_BASE_DIR`
- `python -m pytest tests/ -v`

Observed verification results:

- unauthenticated `POST /api/auth/enable` remained callable after auth was enabled
- authenticated-protected endpoints such as `/api/settings` returned `401` in the same isolated run
- `/api/runtime/bootstrap` returned `401` once auth was enabled
- `pytest` run timed out and then failed in collection due to `tests/test_missing_endpoints.py` calling `sys.exit(1)`

## Open Questions

- Is the product intended to be strictly local-desktop only, or also safe when exposed on a LAN?
- Are update/patch endpoints expected to be used only by trusted desktop UI flows?
- Is anonymous workspace file serving intentional, or an accidental convenience feature?

These answers affect the right fix shape, but they do not change the severity of the current findings.

## Suggested Remediation Order

1. Lock down auth bootstrap, terminal access, and update/patch endpoints.
2. Fix the runtime bootstrap/login contract between backend and frontend.
3. Restrict workspace root changes and static workspace exposure.
4. Repair the pytest suite so the repository has a trustworthy safety net.
5. Clean up config inconsistencies and thread/job lifecycle handling.
