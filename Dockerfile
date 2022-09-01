FROM nvcr.io/nvidia/pytorch:22.08-py3
RUN python -m pip install --upgrade diffusers
RUN python -m pip install --upgrade transformers
WORKDIR /root/.huggingface
COPY token token