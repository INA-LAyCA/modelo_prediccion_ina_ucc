Trabajo Final: 

Modelo de Machine Learning para la predicción del riesgo de Cianobacterias en el Embalse San Roque

Autoras:

- Sofía Cersofios
- Agostina Morellato

Descripción del Proyecto

El Embalse San Roque (ESR), ubicado en Córdoba, Argentina, enfrenta un proceso avanzado de eutrofización que afecta su uso recreativo, su capacidad como fuente de agua potable y la biodiversidad del ecosistema acuático.

Este trabajo desarrolla un sistema predictivo integral basado en Machine Learning para el Instituto Nacional del Agua (INA-SCIRSA), que utiliza bases de datos históricas para:

- Procesar y limpiar datos ambientales.

- Entrenar modelos de predicción de riesgo de proliferación de cianobacterias.

- Servir predicciones a través de una API RESTful.

- Proveer una interfaz gráfica para visualización y consulta de resultados.

Objetivo Final
Este sistema busca optimizar el análisis y monitoreo ambiental del ESR, anticipar eventos de riesgo y brindar herramientas prácticas para la toma de decisiones basada en evidencia científica y tecnológica.

El sistema reemplaza procesos manuales por una arquitectura automatizada, eficiente y robusta que permite:

- Actualización automática de datos mediante triggers y funciones de escucha en la base de datos.

- Entrenamiento y evaluación de modelos predictivos, incluyendo Random Forest y Redes Neuronales.

- Disponibilización de predicciones vía API para consumo desde el frontend.

- Visualización interactiva para investigadores mediante una aplicación en React.

Tecnologías Utilizadas:

Backend y Procesamiento de Datos
- Python 3.12

- Flask – API RESTful

- SQLAlchemy – ORM

- Psycopg2 – Notificaciones de PostgreSQL

- Pandas / NumPy – Limpieza y manipulación de datos

Machine Learning
- Scikit-learn – Modelos clásicos, escalado, imputación

- TensorFlow / Keras – Redes Neuronales

- KerasTuner – Optimización de hiperparámetros

- Joblib – Serialización de modelos

Base de Datos
- PostgreSQL 14.1

- SQL (Triggers y Funciones) – Reactividad ante cambios

Frontend
- React.js

- Axios – Requests HTTP

- React Router

- HTML5 / CSS3

Estructura del Proyecto
backend.py:

1. API Flask con endpoints /datos, /predict, /actualizar.

2. Pipeline de procesamiento obtener_dataframe.

3. Sistema de escucha (LISTEN/NOTIFY) para mantener los datos actualizados.

4. Carga de modelos preentrenados y generación de predicciones.

entrenar_modelos.py:

1. Script offline para entrenamiento y evaluación de modelos.

2. Guarda los artefactos resultantes en /modelos_entrenados.

src/App.js:

1. Componente principal del frontend en React.

2. Gestiona visualización e interacción con el backend.

/modelos_entrenados: Carpeta de modelos .pkl y .keras listos para predicción.

