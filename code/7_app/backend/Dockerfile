# Semblance 2.0 engine — container for Google Cloud Run (see ./README.md, deploy with ./deploy.sh).
# Also runs unchanged on any host that injects $PORT (Hugging Face Spaces, Fly.io, a plain VPS).
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Run as non-root: least privilege, and it keeps the image portable to hosts that require UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
# CPU-only PyTorch first — the default Linux torch wheel drags in ~5 GB of unused CUDA/nvidia
# libraries. Cloud Run is CPU-only (BioLORD runs device="cpu"), so install the CPU build and let
# sentence-transformers see torch already satisfied.
# --upgrade-strategy only-if-needed is pip's default, but it is stated explicitly because it is the
# only thing stopping the second command from pulling the CUDA torch back in over the CPU one.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --upgrade --upgrade-strategy only-if-needed -r requirements.txt

COPY --chown=user . .

# Bake BioLORD into the image so no request ever waits on a ~420 MB download (instance disk is
# ephemeral and scale-to-zero means a fresh instance is the common case). Reactome queries hit the
# shipped reactome_embeddings.npz cache and never touch the model; this is only for non-Reactome
# pathway names. This layer is most of the ~10-15 min first build — it is not hung.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('FremyCompany/BioLORD-2023')"

# Cloud Run injects $PORT (8080). The default below is only for `docker run` locally.
ENV PORT=8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
