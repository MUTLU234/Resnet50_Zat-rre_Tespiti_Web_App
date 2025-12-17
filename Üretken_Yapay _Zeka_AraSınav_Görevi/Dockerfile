# Temel imaj olarak hafif bir Python sürümü kullanıyoruz
FROM python:3.9-slim

# Gerekli sistem kütüphanelerini yükle (OpenCV ve Grad-CAM için)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Gereksinimleri kopyala ve yükle
COPY gereksinimler.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r gereksinimler.txt

# Uygulama kodlarını kopyala
COPY uygulama/ ./uygulama/

# Streamlit varsayılan portu 8501'dir
EXPOSE 8501

# Sağlık kontrolü (Healthcheck) - Konteynerin çalıştığını doğrulamak için
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Uygulamayı başlat
CMD ["streamlit", "run", "uygulama/ana_uygulama.py", "--server.address=0.0.0.0"]
