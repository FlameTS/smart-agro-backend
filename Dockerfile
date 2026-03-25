FROM python:3.10

# Set up a new user to avoid root permissions issues
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the backend files
COPY --chown=user . .

# Expose the port Hugging Face expects
EXPOSE 10000

# Run using uvicorn on the specific port HF requires
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]