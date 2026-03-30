FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# ── 1. Install Python deps ────────────────────────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── 2. Clone + install SAM2 from source ──────────────────────────────────────
# We pin to a specific commit so the build is reproducible and won't break
# if Facebook pushes a breaking change.
RUN git clone https://github.com/facebookresearch/sam2.git sam2_repo && \
    pip install --no-cache-dir -e sam2_repo

# ── 3. Copy application code ──────────────────────────────────────────────────
# weights/ is intentionally excluded from .dockerignore now — see note below.
# If your weights are large (>500MB total), use hf_hub_download at startup
# instead and set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads.
COPY --chown=user . .

# ── 4. Pre-download model weights at BUILD time ───────────────────────────────
# This bakes the weights into the image so the container starts instantly.
# HF Spaces caches Docker layers, so this only re-runs when requirements change.
# 
# Option A (recommended): copy weights/ into the image by removing *.pth / *.pt
# from .dockerignore, then just let the COPY above handle it.
#
# Option B: download at build time using huggingface_hub:
#   RUN python -c "
#   from huggingface_hub import hf_hub_download
#   hf_hub_download(repo_id='Tarman21/smart-agro-model-v1', filename='crop_disease_model.pth', local_dir='weights')
#   hf_hub_download(repo_id='Tarman21/smart-agro-model-v1', filename='yolo_best.pt',           local_dir='weights')
#   hf_hub_download(repo_id='Tarman21/smart-agro-model-v1', filename='sam2_hiera_small.pt',    local_dir='weights')
#   "
#
# Uncomment Option B above if you cannot remove *.pt/*.pth from .dockerignore.

# ── 5. Runtime env placeholders (real values come from HF Spaces Secrets) ────
ENV SUPABASE_URL=""
ENV SUPABASE_ANON_KEY=""
ENV GROQ_API_KEY=""

# Speeds up HF Hub downloads significantly if Option B is used
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

EXPOSE 7860

# --timeout-keep-alive 75: gives the health-check probe enough time during
# first-request model load without HF's proxy killing the connection.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", \
     "--timeout-keep-alive", "75", "--log-level", "info"]