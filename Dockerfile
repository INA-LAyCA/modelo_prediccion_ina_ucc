# Usa una imagen oficial de Python como base. 'slim' es una versión ligera.
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

COPY requirements.txt requirements.txt

# Instala todas las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el resto del código del proyecto al directorio de trabajo
COPY . .

EXPOSE 5001

# El comando para ejecutar la aplicación cuando el contenedor se inicie
CMD ["python3", "backend.py"]