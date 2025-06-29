# --- Etapa 1: Construcción (Build) ---
# Usamos una imagen de Node.js para instalar dependencias y construir la app
FROM node:20 AS builder

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos package.json y package-lock.json para instalar dependencias
COPY package.json package-lock.json ./

# Instalamos las dependencias
RUN npm install

# Copiamos el resto del código fuente de la app
COPY . .

# Construimos la versión de producción de la app
RUN npm run build

# --- Etapa 2: Servicio (Serve) ---
# Usamos una imagen de Nginx súper ligera para servir los archivos estáticos
FROM nginx:alpine

# Copiamos el archivo de configuración personalizado para Nginx
# que crearemos en el siguiente paso.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copiamos los archivos estáticos construidos en la etapa anterior
# al directorio por defecto donde Nginx busca los archivos para servir.
COPY --from=builder /app/build /usr/share/nginx/html

# Exponemos el puerto 80, que es el puerto por defecto de Nginx
EXPOSE 80

# El comando por defecto de la imagen de Nginx iniciará el servidor.
# No necesitamos un CMD.
