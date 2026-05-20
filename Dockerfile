# Image for the Hugging Face Space (Docker SDK). Serves FastAPI + Gradio on 7860.
FROM python:3.11-slim

# Hugging Face Spaces run the container as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Install the CPU build of torch first so the resolver does not pull the
# multi-gigabyte CUDA wheel that the Space cannot use.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# The vector store is rebuilt from the committed corpus parquet on first start.
CMD ["python", "-m", "app.api"]
