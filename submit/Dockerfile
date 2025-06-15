# Gunakan image Python sebagai base image
FROM python:3.9-slim

# Set working directory di container
WORKDIR /code

# Salin requirements dan install dependencies
COPY ./requirements.txt /code/requirements.text

RUN pip install --no-cache-dir -r /code/requirements.text

# Salin seluruh kode aplikasi ke dalam container
COPY ./ /code/app

# Expose port FastAPI
EXPOSE 8000

# Jalankan aplikasi FastAPI menggunakan Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
