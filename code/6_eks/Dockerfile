# Batch image for the CCLE drug-sensitivity fan-out.
# One image, many entry points: the Argo template picks the driver command per shard.
# arm64 to match the Graviton (c7g/m7g) spot nodes and the local `ml` conda env — no cross-build.
#
# Build context is ctrp-gdsc-ccle-ml/ (NOT the repo root — that would tar raw_data/).
#   cd ctrp-gdsc-ccle-ml
#   docker buildx build -f 6_eks/Dockerfile --platform linux/arm64 -t ccle-pipeline:v1 .
#
# The image carries CODE ONLY (the 4_model/*.py drivers). Data is pulled from S3 at run time by
# entrypoint.sh — pods are stateless; S3 is the only data plane.
FROM python:3.11-slim

# Pinned to the versions that run this exact code on the Mac ml env. pyarrow is held at 18.1.0
# on purpose: the pip Linux/arm64 wheel of pyarrow 25.0.0 SEGFAULTS reading our parquet on Graviton
# (crash traced to pyarrow/parquet/core.py). Do NOT let these float — bleeding-edge arm64 wheels bite.
RUN pip install --no-cache-dir awscli \
 && pip install --no-cache-dir \
      numpy==2.4.4 scipy==1.17.1 scikit-learn==1.8.0 pandas==3.0.2 pyarrow==18.1.0 joblib==1.5.3

# just the modeling drivers (self-contained: data.py/evaluate.py/models.py/run_*.py/emit_shards.py)
COPY 4_model/*.py /app/ctrp-gdsc-ccle-ml/4_model/
COPY 6_eks/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
