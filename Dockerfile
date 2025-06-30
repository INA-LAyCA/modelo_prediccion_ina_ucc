# Usa una imagen oficial de Python como base. 'slim' es una versión ligera.
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia primero el archivo de dependencias.
# Esto aprovecha el caché de Docker: si no cambias las dependencias,
# este paso no se vuelve a ejecutar, haciendo las construcciones más rápidas.
COPY requirements.txt requirements.txt

# Instala todas las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el resto del código del proyecto al directorio de trabajo
COPY . .


# Expone el puerto en el que Flask se ejecuta dentro del contenedor
EXPOSE 5001

# El comando para ejecutar la aplicación cuando el contenedor se inicie
CMD ["python", "backend.py"]