# Taiji 2.0 Architecture Consolidation

Taiji 2.0 is not a feature expansion plan. It is a consolidation plan for
startup ownership, runtime state, API orchestration, domain boundaries, and the
client shell.

## Diagnosis

The current project can run through multiple paths, but those paths do not share
one architectural contract:

- Desktop, API, dev server, and static SPA entry points all behave like primary
  entries.
- Runtime state is split across backend `app_state`, Pinia stores, page-local
  requests, and `localStorage`.
- Route modules directly orchestrate Agent, Life, Training, Model, Tool, and Auth
  objects.
- Core concepts are repeated across `taiji.agent`, `taiji.agent_ext`,
  `taiji.life`, `taiji.body`, `taiji.core`, and `taiji.model_ext`.
- The frontend is still a set of pages around a sidebar, not a product-grade
  client shell with one view of runtime, auth, errors, tools, and model lifecycle.
- CSS ownership is split across global app CSS, layout CSS, overrides, and
  component scoped styles.

## Target Shape

### 1. One Product Entry

Primary product entry:

- `desktop/main.py` starts the packaged desktop client.
- The desktop client starts the API runtime and loads the built frontend from
  the API server.

Development entries:

- `frontend` Vite dev server is a frontend development entry only.
- API server CLI entry is a backend development and test entry only.
- Static `dist` is a deployment artifact, not an independent runtime owner.

Compatibility entries:

- Existing launch scripts may remain, but should delegate to the primary product
  or development entry instead of duplicating boot logic.

### 2. Runtime Center

The client shell reads runtime state from:

```text
GET /api/runtime/status
```

This endpoint is the frontend read model for:

- runtime health and model lifecycle
- startup download status
- memory pressure
- auth status
- life state
- tool availability
- training and publishing state

Routes must not rebuild this payload locally. They should call the runtime
service read model.

### 3. Domain Services

API routes are transport adapters. Business orchestration belongs in services:

```text
taiji/services/runtime_service.py
taiji/services/agent_service.py
taiji/services/life_service.py
taiji/services/model_service.py
taiji/services/training_service.py
taiji/services/tool_service.py
taiji/services/auth_service.py
```

Service rules:

- Services may coordinate domain modules.
- Services may read or update `app_state` through explicit methods.
- Routes should validate request shape, call one service method, and serialize
  the result.
- Domain modules should not import API modules.

### 4. Client Shell

The frontend shell owns global product state:

- runtime connection and model lifecycle
- auth requirement and login state
- exception center
- active tools and tool categories
- terminal status
- global navigation
- layout density and responsive behavior

Pages should render workflow content only. They should not independently poll
runtime, auth, memory, life, or tools.

### 5. Design System Exit Plan

CSS ownership should move toward:

```text
frontend/src/assets/styles/tokens.css
frontend/src/assets/styles/reset.css
frontend/src/assets/styles/shell.css
frontend/src/assets/styles/components.css
```

Migration rules:

- Tokens define color, radius, spacing, shadow, typography, and status colors.
- Shell CSS owns app layout only.
- Component CSS owns reusable primitives only.
- Page styles stay local and must not override shell or primitive behavior.
- `overrides.css` should shrink over time and eventually disappear.

## Phase Plan

### Phase 0: Contract Freeze

- Document product, dev, and compatibility entries.
- Keep old entry files working, but mark non-primary entries as delegates.
- Add `/api/runtime/status` as the single frontend status contract.
- Add tests for the runtime payload shape.

### Phase 1: Runtime First

- Move runtime aggregation into `taiji.services.runtime_service`.
- Make `/api/runtime/status` a thin adapter.
- Make frontend `runtimeStore` consume the runtime payload first.
- Keep legacy endpoints for compatibility while pages migrate.

### Phase 2: Service Extraction

Extract orchestration from routes in this order:

1. Tools, because Agent routes and Runtime Center both need them.
2. Auth, because shell boot depends on it.
3. Model lifecycle, because startup, switching, health, and chat depend on it.
4. Training, because it has long-running state and locks.
5. Life, because it currently spans scheduler, needs, body, and UI expression.
6. Agent, because it combines memory, tools, ReAct, workspace, and MCP.

### Phase 3: Client Shell

- Make `App.vue` only compose providers, shell, and router outlet.
- Move global runtime display into shell components.
- Pages receive runtime state from `runtimeStore`, not direct polling.
- Remove page-level localStorage ownership except user preferences.

### Phase 4: Core Boundary Cleanup

Normalize domain ownership:

- `taiji.core`: runtime infrastructure, config, app state, loading, security.
- `taiji.agent`: stable agent domain interfaces.
- `taiji.agent_ext`: experimental or plugin-like extensions until promoted.
- `taiji.life`: life scheduler, needs, evolution, and life-facing services.
- `taiji.body`: embodied IO abstractions only, not duplicated life state.
- `taiji.model_ext`: model registry, download, training utilities, GGUF support.
- `taiji.tools`: callable tool implementations.

### Phase 5: Design System Migration

- Freeze new additions to `overrides.css`.
- Move repeated colors and spacing into tokens.
- Replace page-specific layout workarounds with shell primitives.
- Delete dead global selectors after each page migration.

## First Cut Implemented

This consolidation starts with the runtime path:

- Added `taiji/services/runtime_service.py`.
- Reduced `api/routes_runtime.py` to an API adapter.
- Reused the runtime tool status from `/api/agent/tools`.
- Changed the frontend runtime store to synchronize from
  `/api/runtime/status`.
- Changed frontend health polling to use `/api/runtime/status` instead of
  `/api/health`.

The old health and domain endpoints remain available for compatibility while the
client shell migration continues.
