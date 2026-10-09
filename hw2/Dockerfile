FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel AS base

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip \
    && python3 -m pip install 'transformers==4.56.2' 'datasets==3.6.0' \
        'accelerate==1.6.0' 'trl==0.23.1' 'liger-kernel==0.6.2' 'jupyterlab==4.4.9'

RUN python3 -m pip install --no-cache-dir --no-build-isolation \
    'flash-attn==2.7.3'

ENV HF_DISABLE_TELEMETRY=true \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8890 \
    PROJECT_ROOT=/app

ENV PATH="$PROJECT_ROOT/bin:$PATH" PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

COPY train.py communications.ipynb ./
