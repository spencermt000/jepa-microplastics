FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace
ENV MLFLOW_TRACKING_URI=http://192.168.4.51:5000

ENTRYPOINT ["python"]
CMD ["scripts/pretrain.py", "--config", "configs/pretrain.yaml"]
