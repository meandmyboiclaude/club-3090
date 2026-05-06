#!/usr/bin/env bash
# Bootstraps ComfyUI + custom nodes into the mounted /workspace/ComfyUI volume on first run,
# pulls latest on subsequent runs, then launches the ComfyUI server on :8188.
set -euo pipefail

COMFY_ROOT="/workspace/ComfyUI"
PIP_OPTS="--no-input --retries 10 --timeout 60"

clone_or_update() {
    local url="$1"
    local dest="$2"
    if [ ! -d "$dest/.git" ]; then
        echo "[bootstrap] git clone $url -> $dest"
        git clone --depth 1 "$url" "$dest"
    else
        echo "[bootstrap] git pull $dest"
        git -C "$dest" fetch --depth 1 origin || true
        git -C "$dest" reset --hard "@{u}" 2>/dev/null || git -C "$dest" pull --ff-only || true
    fi
}

# 0. Make models/input/output/user dirs writable by the host user (uid 1000) too,
#    so host-side hf download / file moves work after container has touched them as root.
chmod -R a+rwX /workspace/ComfyUI/models /workspace/ComfyUI/input /workspace/ComfyUI/output /workspace/ComfyUI/user 2>/dev/null || true

# 1. ComfyUI core — clone into a tmp dir then sync over (target dir is non-empty due to bind mounts)
if [ ! -f "$COMFY_ROOT/main.py" ]; then
    echo "[bootstrap] Cloning ComfyUI into temp..."
    rm -rf /tmp/_comfy_clone
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /tmp/_comfy_clone
    mkdir -p "$COMFY_ROOT"
    # Move everything except dirs that exist as bind mounts (models, input, output, user)
    cp -an /tmp/_comfy_clone/. "$COMFY_ROOT/"
    rm -rf /tmp/_comfy_clone
elif [ -d "$COMFY_ROOT/.git" ]; then
    echo "[bootstrap] ComfyUI already present, pulling..."
    git -C "$COMFY_ROOT" pull --ff-only || true
else
    echo "[bootstrap] ComfyUI present without .git; skipping pull."
fi

cd "$COMFY_ROOT"

# 2. Core requirements (don't fail container if a transient pull fails — already-installed deps stay)
echo "[bootstrap] Installing ComfyUI core requirements..."
pip install $PIP_OPTS -r requirements.txt || echo "[bootstrap] WARN: core pip install had errors; continuing"

# 3. Custom nodes
NODES="$COMFY_ROOT/custom_nodes"
mkdir -p "$NODES"

clone_or_update https://github.com/Comfy-Org/ComfyUI-Manager.git "$NODES/ComfyUI-Manager"
clone_or_update https://github.com/city96/ComfyUI-GGUF.git           "$NODES/ComfyUI-GGUF"
clone_or_update https://github.com/mit-han-lab/ComfyUI-nunchaku.git  "$NODES/ComfyUI-nunchaku"
clone_or_update https://github.com/kijai/ComfyUI-WanVideoWrapper.git "$NODES/ComfyUI-WanVideoWrapper"
clone_or_update https://github.com/kijai/ComfyUI-HunyuanVideoWrapper.git "$NODES/ComfyUI-HunyuanVideoWrapper"
clone_or_update https://github.com/kijai/ComfyUI-KJNodes.git         "$NODES/ComfyUI-KJNodes"
clone_or_update https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "$NODES/ComfyUI-VideoHelperSuite"
clone_or_update https://github.com/pollockjj/ComfyUI-MultiGPU.git "$NODES/ComfyUI-MultiGPU"

for d in "$NODES"/*/; do
    if [ -f "$d/requirements.txt" ]; then
        echo "[bootstrap] Installing requirements for $(basename "$d")..."
        pip install $PIP_OPTS -r "$d/requirements.txt" || echo "[bootstrap] WARN: requirements failed for $(basename "$d")"
    fi
done

# 4. Nunchaku — pin v1.2.0 wheel for torch 2.7 + cu12.8 (sm_86 supported).
#    NOTE: PyPI 'nunchaku' is a different stats package; do not pip install nunchaku without the URL.
NUNCHAKU_WHL="https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.0/nunchaku-1.2.0+torch2.7-cp311-cp311-linux_x86_64.whl"
if ! python3 -c "import nunchaku; assert nunchaku.__file__.find('site-packages/nunchaku/') >= 0" 2>/dev/null; then
    echo "[bootstrap] Installing nunchaku 1.2.0 from prebuilt wheel..."
    pip uninstall -y nunchaku 2>/dev/null || true
    pip install $PIP_OPTS "$NUNCHAKU_WHL" || echo "[bootstrap] WARN: nunchaku wheel install failed"
else
    echo "[bootstrap] nunchaku already installed."
fi

# 5. Launch ComfyUI
echo "[bootstrap] Starting ComfyUI on 0.0.0.0:8188"
exec python main.py --listen 0.0.0.0 --port 8188 "$@"
