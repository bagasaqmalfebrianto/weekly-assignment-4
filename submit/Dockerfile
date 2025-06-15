# Gunakan image Python sebagai base image
FROM python:3.9-slim

# Set working directory di container
WORKDIR /app

# Salin requirements dan install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY . .

# Expose port FastAPI
EXPOSE 8000

# Jalankan aplikasi FastAPI menggunakan Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
