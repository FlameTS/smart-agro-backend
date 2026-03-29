FROM python:3.10-slim

# System dependencies
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

# Install Python dependencies first (better Docker layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install SAM2 (before COPY to leverage caching)
RUN git clone https://github.com/facebookresearch/sam2.git sam2_repo
RUN pip install --no-cache-dir -e sam2_repo

# Copy rest of project (.dockerignore excludes sam2_repo/, .env, etc.)
COPY --chown=user . .

# HF Spaces injects secrets as env vars at runtime — these are fallback defaults
ENV SUPABASE_URL=""
ENV SUPABASE_ANON_KEY=""
ENV GROQ_API_KEY=""

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]