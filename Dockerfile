# Official lightweight PyTorch image with CUDA 12.4 and cuDNN 9 runtime
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache IndoBERT weights into image so training starts immediately
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1'); \
    AutoModel.from_pretrained('indobenchmark/indobert-base-p1')"

# Copy source code, configs, and processed dataset
COPY . .

# Ensure directory for saved models
RUN mkdir -p checkpoints

# Default entrypoint and training command
ENTRYPOINT ["python", "train.py"]
CMD ["tracker=wandb", "model.pretrained_model=indobenchmark/indobert-base-p1"]