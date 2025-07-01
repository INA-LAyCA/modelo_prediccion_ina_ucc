
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

FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/build /usr/share/nginx/html

# Exponemos el puerto 80, que es el puerto por defecto de Nginx
EXPOSE 80

