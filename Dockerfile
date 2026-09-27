# CrediX Multi-Platform Container Deployment
# Compatible with Render, Railway, Hugging Face Spaces, Google Cloud Run, AWS ECS, Fly.io
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# Install git and git-lfs for large models
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code and assets
COPY . .

# Expose standard Streamlit port
EXPOSE 8501

# Healthcheck for uptime monitors and orchestrators
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Streamlit command configured to run without headless prompts and bind to 0.0.0.0
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
