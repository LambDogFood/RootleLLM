# GPU training image for rootllm.
#
# The base image pins CUDA + PyTorch, so you skip the manual Blackwell/sm_120
# install dance. Pick a base with PyTorch >= 2.7 / CUDA >= 12.8 for the RTX 5070.
# If the Docker Hub tag lags on Blackwell support, use NVIDIA's NGC image instead:
#   --build-arg BASE=nvcr.io/nvidia/pytorch:25.01-py3
ARG BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE}

WORKDIR /workspace/rootllm

# Project deps only — torch already ships in the base image, so don't let pip
# pull a second (possibly mismatched) copy. Hence --no-deps below.
RUN pip install --no-cache-dir numpy pyyaml tiktoken

COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# data/ and out/ are meant to be bind-mounted so they persist across runs.
ENV PYTHONPATH=/workspace/rootllm/src
ENTRYPOINT []
CMD ["python", "-m", "rootllm"]
