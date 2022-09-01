## Docker `RUN` Instructions
Use the following command to run the sample script and place the images here in this directory
```bash
docker run --gpus all -it --rm -v $(pwd):/app huggingface-diffuser:20220828 python /app/diffuser.py
```