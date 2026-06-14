# Image for the QLoRA engine (pretrained code model + LoRA). Same CUDA base as the
# from-scratch image, plus the HuggingFace stack. For Blackwell (5070) use a base
# with PyTorch >= 2.7 / CUDA 12.8; switch to the NGC image if needed.
ARG BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE}

WORKDIR /workspace/rootllm

COPY qlora/requirements.txt qlora/requirements.txt
RUN pip install --no-cache-dir -r qlora/requirements.txt

COPY . .
ENV PYTHONPATH=/workspace/rootllm/src
ENTRYPOINT []
CMD ["python", "qlora/finetune.py"]
