FROM python:3.10

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM2 from bundled repo
COPY --chown=user sam2_repo/ ./sam2_repo/
RUN pip install --no-cache-dir -e ./sam2_repo

# Copy rest of project
COPY --chown=user . .

ENV SUPABASE_URL=""
ENV SUPABASE_ANON_KEY=""

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]