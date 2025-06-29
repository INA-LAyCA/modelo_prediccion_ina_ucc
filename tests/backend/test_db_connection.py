# tests/backend/test_db_connection.py
import pytest
from sqlalchemy import create_engine, text
import pandas as pd

# Marcar estas pruebas como 'db' para poder ejecutarlas por separado si se desea
# Estas pruebas solo funcionarán si el contenedor de Docker está corriendo
@pytest.mark.db
def test_database_connection_and_query():
    """
    Prueba una conexión real a la BD de prueba y una consulta simple.
    """
    # Conexión a la base de datos de prueba levantada con Docker
    db_url = "postgresql+psycopg2://testuser:testpass@localhost:54321/testdb"
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            # 1. Crear una tabla de prueba y insertar datos
            connection.execute(text("DROP TABLE IF EXISTS prueba;"))
            connection.execute(text("CREATE TABLE prueba (id INT, nombre VARCHAR(50));"))
            connection.execute(text("INSERT INTO prueba (id, nombre) VALUES (1, 'hola');"))
            connection.commit()

            # 2. Leer los datos usando pandas
            df = pd.read_sql("SELECT * FROM prueba;", connection)

            # 3. Verificar los datos
            assert not df.empty
            assert df.loc[0, 'nombre'] == 'hola'

            # 4. Limpiar
            connection.execute(text("DROP TABLE prueba;"))
            connection.commit()

    except Exception as e:
        pytest.fail(f"No se pudo conectar a la base de datos de prueba en {db_url}. "
                    f"Asegúrate de que el contenedor de Docker esté corriendo. Error: {e}")