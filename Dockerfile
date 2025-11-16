# Temel Python imajı
FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# MODELİ DOĞRU YERE KOPYALA
COPY model_cache/models--runwayml--stable-diffusion-v1-5 /root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
