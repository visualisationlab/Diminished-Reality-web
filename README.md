# Diminished Reality

Mobile web app that detects grocery items in real time and visually de-emphasizes unhealthy ones with AR effects.

## Requirements

- `pnpm`
- `uv`
- `Python 3.11+`
- `git-lfs`

## Install

```bash
git lfs install

cd frontend
pnpm install

cd ../backend
uv sync
```

The ONNX model is tracked in Git LFS at `backend/model/model.onnx`.

After cloning, fetch LFS objects if the model has not been downloaded yet:

```bash
git lfs pull
```

## Run

Linux/macOS:

```bash
./start-backend.sh
./start-frontend.sh
```

Windows:

```bat
start-backend.bat
start-frontend.bat
```

Frontend runs on `https://localhost:5173` and proxies `/ws` to the backend on `ws://localhost:5174`.
