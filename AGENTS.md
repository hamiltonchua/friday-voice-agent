# AGENTS.md — Kismet Voice Agent

> Real-time voice chat interface for OpenClaw agents. Python FastAPI backend + React/TypeScript frontend.

## Architecture

Two-part application (NOT a monorepo):
- **Backend** — Python FastAPI server (`server.py`) with WebSocket-based voice pipeline
  - `server.py` — main server: WebSocket handler, STT/TTS/LLM pipeline, audio streaming
  - `platform_config.py` — auto-detects hardware (MLX/CUDA/CPU), sets backend defaults
  - `auth.py` — WebAuthn (passkey/Touch ID) authentication, session cookies
  - `speaker_verify.py` — SpeechBrain ECAPA-TDNN speaker verification
- **Frontend** — React 19 + TypeScript + Vite app in `frontend/`
  - Built to `frontend/dist/`, served by the Python backend
  - Communicates with backend via WebSocket at `/ws`

## Build / Lint / Test Commands

### Frontend (run from `frontend/`)

```bash
npm install            # Install dependencies
npm run dev            # Vite dev server (proxies /ws to localhost:8765)
npm run build          # TypeScript check + Vite production build (tsc -b && vite build)
npm run lint           # ESLint (flat config, TS/TSX only)
npm run preview        # Preview production build
```

### Backend (Python)

```bash
conda activate voice-agent           # Activate conda environment
pip install -r requirements-macos.txt  # macOS Apple Silicon
pip install -r requirements-linux.txt  # Linux CUDA
pip install -r requirements.txt        # Generic/CUDA fallback

python server.py                       # Start server on https://0.0.0.0:8765
./start-kokoro.sh                      # Quick start macOS (Kokoro TTS)
./start-chatterbox.sh                  # Quick start Linux (Chatterbox TTS)
```

### Tests

No test suite exists. No test framework is configured (no pytest, vitest, jest).
When adding tests: use `pytest` for Python, `vitest` for frontend.

### Type Checking

```bash
# Frontend — strict mode enabled, runs as part of `npm run build`
npx tsc -b                    # Standalone type check (from frontend/)
```

TypeScript strict settings: `noUnusedLocals`, `noUnusedParameters`, `erasableSyntaxOnly`,
`noFallthroughCasesInSwitch`, `verbatimModuleSyntax`, `noUncheckedSideEffectImports`.

## Frontend Code Style

### Imports

Order (no enforced linter rule, follow existing convention):
1. React hooks (`useState`, `useRef`, `useCallback`, `useEffect`)
2. External libraries (`sonner`, `lucide-react`, `@ricky0123/vad-web`)
3. Local hooks (`./hooks/useWebSocket`, etc.)
4. Constants and types (`./constants`, `./types`)
5. Components (`./components/Header`, etc.)

Use `type` keyword for type-only imports: `import type { ChatMessage } from './types'`
Path alias: `@/*` maps to `src/*` (configured in tsconfig + vite). Use for deep imports.

### Formatting

- **No semicolons** in application code (shadcn UI files may have them — that's fine)
- **Single quotes** for strings
- **No formatter configured** (no Prettier/Biome) — match surrounding code style
- **2-space indentation** (TypeScript and Python)

### Components

- One component per file in `src/components/`
- **Named exports**: `export function Header() {}` — not default exports
  - Exception: `App.tsx` uses `export default function App()`
- Props interface defined directly above the component in the same file
- Shadcn UI components live in `src/components/ui/` (auto-generated, new-york style)
- Icons from `lucide-react`

### Hooks

- Custom hooks in `src/hooks/`, one per file
- Return objects or tuples
- Use `useRef` for mutable state that should not trigger re-renders
- Wrap callbacks in `useCallback`

### Types

- Shared types in `src/types.ts`, constants in `src/constants.ts`
- Use TypeScript interfaces for component props and data shapes
- Use union string literals for enums: `type ClientState = 'sleeping' | 'awake'`
- No `any` — use `unknown` with type narrowing or `Record<string, unknown>`

### State Management

- All application state lives in `App.tsx` via `useState`/`useRef`
- No external state library (no Redux, Zustand, etc.)
- Props drilling is the pattern — pass state and callbacks down as props

### Error Handling (Frontend)

- `try/catch` with `err as Error` type assertion
- User-facing errors via `toast.error()` from `sonner`
- Console logging with `[WS]`, `[VAD]`, etc. prefixes

### CSS / Styling

- Tailwind CSS v4 (via `@tailwindcss/vite` plugin)
- CSS custom properties in `src/index.css`
- Mix of Tailwind utility classes and inline `style={{}}` objects
- Shadcn UI uses `class-variance-authority`, `clsx`, `tailwind-merge` via `cn()` util

## Backend Code Style (Python)

### Naming

- `snake_case` for functions and variables
- `UPPER_SNAKE_CASE` for constants and env vars
- `PascalCase` for classes
- Private helpers prefixed with `_` (e.g., `_detect_platform()`, `_read_openclaw_token()`)

### Imports

- Standard library first, then third-party, then local modules
- Lazy imports inside functions for heavy ML libraries (avoid startup cost):
  ```python
  def get_whisper():
      from mlx_audio.stt.utils import load_model  # lazy
  ```

### Type Hints

- Used selectively, not exhaustively
- Modern Python 3.10+ syntax: `str | None`, `tuple[bool, float]`, `list[dict]`
- `Optional[T]` and `AsyncIterator` from `typing` for older-style hints

### Error Handling (Backend)

- `try/except` with specific exception types
- Print-based logging with module prefixes: `[STT]`, `[TTS]`, `[WakeWord]`, `[Auth]`
- No structured logging framework

### File Organization

- Section dividers: `# ---------------------------------------------------------------------------`
- Constants at top, then global singletons, then functions, then FastAPI routes, then `__main__`
- Thread-safe model loading with `threading.Lock` and `global` keyword
- Module-level docstrings with triple quotes

## Environment

- Python 3.11+ required
- Node.js 18+ required for frontend
- conda recommended for Python env (`voice-agent`)
- Server runs on port 8765 with HTTPS (self-signed certs)
- No CI/CD pipeline configured
- No `.env.example` — env vars documented in README.md tables

## Key Env Vars

| Variable | Default | Purpose |
|---|---|---|
| `KISMET_PLATFORM` | auto-detect | Force `mlx`, `cuda`, or `cpu` |
| `OPENCLAW_URL` | `http://127.0.0.1:18789/v1/chat/completions` | LLM endpoint |
| `OPENCLAW_TOKEN` | — | Gateway auth token |
| `OPENCLAW_AGENT` | `main` | Agent ID |
| `STT_BACKEND` | auto | `mlx-audio` or `faster-whisper` |
| `TTS_BACKEND` | auto | `mlx-audio`, `chatterbox-cuda`, or `kokoro-onnx` |
| `WAKE_WORD_ENABLED` | `true` | Enable wake word detection |
| `SPEAKER_VERIFY` | `auto` | Speaker verification mode |

## Git Conventions

- Commit style: imperative mood, optional `feat:` / `fix:` prefix (not strictly enforced)
- Do not commit: `.env`, `*.ppn`, `*.onnx`, `*.bin`, `auth_credentials.json`, `cert.pem`, `key.pem`
