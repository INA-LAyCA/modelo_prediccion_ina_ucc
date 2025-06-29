# tests/backend/test_backend.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from backend import obtener_dataframe, actualizar_df
import time
import threading
import psycopg2
import pytest
from backend import database_listener

from backend import (
    asignar_estacion,
    imputar_cota_m,
    seleccionar_medicion_mensual,
    union_precipitacion,
    imputacion_clorofila,
    imputar_pht,
    imputar_prs,
    imputar_nitrogeno,
    imputacion_temperatura_aire,
    union_temperatura_aire,
    imputacion_temperatura_agua,
    condicion_termica
)

def test_asignar_estacion():
    """Prueba la lógica de asignación de estaciones."""
    assert asignar_estacion(1) == 'Verano'   # Enero
    assert asignar_estacion(4) == 'Otoño'    # Abril
    assert asignar_estacion(7) == 'Invierno' # Julio
    assert asignar_estacion(11) == 'Primavera' # Noviembre
    assert asignar_estacion(12) == 'Verano'  # Diciembre

def test_imputar_cota_m():
    """Prueba la imputación de la columna 'Cota (m)'."""
    data = {
        'fecha': pd.to_datetime(['2023-01-10', '2023-01-10', '2023-01-15']),
        'Cota (m)': [100.0, np.nan, 102.0]
    }
    df_input = pd.DataFrame(data)
    df_output = imputar_cota_m(df_input)

    # El NaN para la misma fecha debería ser rellenado con el valor anterior/posterior (100.0)
    assert df_output['Cota (m)'].isna().sum() == 0
    assert df_output.loc[1, 'Cota (m)'] == 100.0

def test_seleccionar_medicion_mensual():
    """Prueba la selección de la medición más representativa del mes."""
    data = {
        'fecha': pd.to_datetime(['2023-01-10', '2023-01-20', '2023-01-25']),
        'codigo_perfil': ['C1', 'C1', 'C1'],
        'id_registro': [1, 2, 3],
        # La medición del día 20 tiene menos faltantes, debería ser la elegida
        'Clorofila (µg/l)': [np.nan, 20.0, np.nan],
        'T° (°C)': [15.0, 16.0, 17.0]
    }
    df_input = pd.DataFrame(data)
    df_output = seleccionar_medicion_mensual(df_input)

    # El resultado debería tener una sola fila para el perfil C1 en el mes de Enero
    assert len(df_output) == 1
    # Y debería ser la del id_registro 2, que es la más completa y reciente
    assert df_output.iloc[0]['id_registro'] == 2

def test_union_precipitacion(mocker):
    """
    Prueba que la unión con los datos de precipitación funcione correctamente.
    """
    # 1. Datos de entrada
    df_final_input = pd.DataFrame({
        'fecha': [datetime(2023, 10, 20).date(), datetime(2023, 10, 21).date()]
    })
    
    # 2. Simular lo que la base de datos devolvería
    mock_df_precipitacion = pd.DataFrame({
        'fecha_dia': pd.to_datetime(['2023-10-20', '2023-10-21']),
        'sensor_id': [600, 700],
        'precipitacion_acumulada_3d': [10.5, 5.2]
    })
    mocker.patch('pandas.read_sql', return_value=mock_df_precipitacion)
    
    # 3. Llamar a la función
    # La función necesita un 'engine' simulado, pero no se usará gracias al mock
    df_output = union_precipitacion(df_final_input, engine2=None) 
    
    # 4. Verificar el resultado
    assert 600 in df_output.columns
    assert 700 in df_output.columns
    assert df_output.loc[0, 600] == 10.5
    assert pd.isna(df_output.loc[0, 700]) # Verifica que el merge 'left' funcionó

def test_union_temperatura_aire(mocker):
    """
    Prueba que la unión con los datos de temperatura del aire funcione.
    """
    # 1. Datos de entrada
    df_final_input = pd.DataFrame({
        'fecha': pd.to_datetime(['2023-11-01', '2023-11-02'])
    })
    
    # 2. Simular la respuesta de la BD
    mock_df_temp = pd.DataFrame({
        'fecha_dia': pd.to_datetime(['2023-11-01']),
        'temperatura_max': [25.0],
        'temperatura_min': [15.0]
    })
    mocker.patch('pandas.read_sql', return_value=mock_df_temp)

    # 3. Llamar a la función
    df_output = union_temperatura_aire(df_final_input, engine2=None)

    # 4. Verificar
    assert 'temperatura_max' in df_output.columns
    assert df_output.loc[0, 'temperatura_max'] == 25.0
    assert pd.isna(df_output.loc[1, 'temperatura_max']) # El día 2 no tenía datos

# --- Pruebas para Funciones de Imputación ---

def test_imputacion_temperatura_agua():
    """Prueba la imputación por media de grupo para la temperatura del agua."""
    df_input = pd.DataFrame({
        'codigo_perfil': ['C1', 'C1', 'C1'],
        'estacion': ['Verano', 'Verano', 'Verano'],
        'T° (°C)': [25.0, 27.0, np.nan]
    })
    df_output = imputacion_temperatura_agua(df_input)
    assert df_output['T° (°C)'].isna().sum() == 0
    assert df_output.loc[2, 'T° (°C)'] == 26.0 # La media de 25 y 27

def test_imputacion_temperatura_aire():
    """Prueba la imputación por media para la temperatura del aire."""
    df_input = pd.DataFrame({
        'estacion': ['Invierno', 'Invierno'],
        'mes': [7, 7],
        'temperatura_max': [15.0, np.nan],
        'temperatura_min': [np.nan, 5.0]
    })
    df_output = imputacion_temperatura_aire(df_input)
    assert df_output['temperatura_max'].notna().all()
    assert df_output['temperatura_min'].notna().all()
    assert df_output.loc[1, 'temperatura_max'] == 15.0 # Media del grupo
    assert df_output.loc[0, 'temperatura_min'] == 5.0  # Media del grupo

def test_imputacion_clorofila_usa_mediana():
    """
    Prueba que la imputación de clorofila use la mediana cuando el modelo no es mejor.
    """
    df_input = pd.DataFrame({
        'id_registro': range(6),
        'codigo_perfil': ['C1'] * 6,
        'estacion': ['Verano'] * 6,
        'Total Algas Sumatoria (Cel/L)': [1000, 1100, 1200, 1300, 1400, 1050],
        'Cianobacterias Total': [500, 550, 600, 650, 700, 525],
        'T° (°C)': [22, 23, 24, 25, 26, 21],
        'Clorofila (µg/l)': [10, 12, 50, 13, 11, np.nan], # El 50 es un outlier
        # --- AÑADIR ESTAS LÍNEAS ---
        'PHT (µg/l)': [np.nan] * 6,
        'PRS (µg/l)': [np.nan] * 6,
    })
    # Con el outlier, la mediana (~11.5) será mucho mejor predictor que un modelo lineal
    df_output, resultados = imputacion_clorofila(df_input)
    
    assert pd.notna(df_output.loc[5, 'Clorofila (µg/l)'])
    assert abs(df_output.loc[5, 'Clorofila (µg/l)'] - 11.5) < 1 
    assert resultados.iloc[0]['metodo'] == 'mediana'


def test_imputar_pht_y_prs():
    """
    Prueba un caso simple para imputar PHT y PRS.
    """
    df_input = pd.DataFrame({
        'id_registro': range(6),
        'codigo_perfil': ['C1'] * 6,
        'estacion': ['Verano'] * 6,
        'Clorofila (µg/l)': [10, 11, 12, 13, 14, 15],
        'Total Algas Sumatoria (Cel/L)': [1000, 1100, 1200, 1300, 1400, 1050],
        'Cianobacterias Total': [500, 550, 600, 650, 700, 525],
        'T° (°C)': [22, 23, 24, 25, 26, 21],
        'PHT (µg/l)': [100, 110, 120, 130, 140, np.nan],
        'PRS (µg/l)': [50, 55, 60, 65, 70, np.nan]
    })
    
    # Probar PHT
    df_output_pht, _ = imputar_pht(df_input)
    assert pd.notna(df_output_pht.loc[5, 'PHT (µg/l)'])
    
    # Probar PRS (usando el output que ya tiene PHT imputado)
    df_output_prs, _ = imputar_prs(df_output_pht)
    assert pd.notna(df_output_prs.loc[5, 'PRS (µg/l)'])

# --- Prueba para una función compleja con lógica condicional ---

def test_condicion_termica_calculo(mocker):
    """
    Prueba que la función `condicion_termica` calcule 'ESTRATIFICADA'
    cuando los datos de entrada lo indican.
    """
    # 1. Datos de entrada para la función
    df_principal = pd.DataFrame({
        'id_registro': [101, 102],
        'condicion_termica': [np.nan, np.nan],
        # ... otras columnas ...
    })
    
    # 2. Simular lo que la BD devolvería para `vista_condicion_termica`
    # Datos que fuerzan la condición 'ESTRATIFICADA' (dif T >= 1.0 en dif z <= 1)
    mock_df_ct = pd.DataFrame({
        'id_registro': [101, 102],
        'fecha': [datetime(2023, 1, 5), datetime(2023, 1, 5)],
        'codigo_perfil': ['C1', 'C1'],
        'parametro': ['T° (°C)', 'T° (°C)'],
        'valor': [25.0, 23.5], # dif T = 1.5
        'z': [1.0, 2.0],       # dif z = 1.0
        'condicion_termica': [np.nan, np.nan]
    })
    mocker.patch('pandas.read_sql', return_value=mock_df_ct)

    # 3. Llamar a la función
    df_output = condicion_termica(df_principal, db_engine_param=None)

    # 4. Verificar
    # La función debe haber actualizado la condición térmica en el df principal
    assert df_output.loc[0, 'condicion_termica'] == 'ESTRATIFICADA'


def test_obtener_dataframe_pipeline_completo(mocker):
    """
    Prueba el pipeline completo de obtener_dataframe simulando las lecturas de la BD.
    """
    # 1. Crear DataFrames simulados que pd.read_sql devolverá
    mock_df_vistaconjunto = pd.DataFrame({
        # 7 filas en total para coincidir con los 7 parámetros
        'id_registro': [1] * 5 + [2] * 2,
        'condicion_termica': [None] * 5 + ['MEZCLA'] * 2,
        'fecha': [datetime(2023, 5, 15)] * 5 + [datetime(2023, 6, 20)] * 2,
        'codigo_perfil': ['C1'] * 7,
        'descripcion_estratificacion': [None] * 7,
        'parametro': [
            'T° (°C)',
            'Anabaena',
            'Total Algas Sumatoria (Cel/L)',
            'PHT (µg/l)',
            'PRS (µg/l)',
            'Clorofila (µg/l)',
            'Microcystis'
        ],
        'valor_parametro': [
            22.5,
            50.0,
            1500.0,
            10.0,   # <--- CAMBIO CLAVE: De np.nan a un número
            5.0,    # <--- CAMBIO CLAVE: De np.nan a un número
            15.0,
            100.0
        ]
    })

    # Simula las otras vistas que se leen
    mock_df_vista_alerts = pd.DataFrame()
    mock_df_vista_precipitacion = pd.DataFrame()
    mock_df_vista_temperatura = pd.DataFrame({
        'fecha_dia': [datetime(2023, 5, 15)],
        'temperatura_max': [25.0],
        'temperatura_min': [15.0]
    })
    mock_df_vista_condicion_termica = pd.DataFrame()

    # 2. Configurar el mock para que devuelva los DataFrames en el orden correcto
    mocker.patch('pandas.read_sql', side_effect=[
        mock_df_vistaconjunto,
        mock_df_vista_alerts,
        mock_df_vista_condicion_termica,
        mock_df_vista_temperatura,
        mock_df_vista_precipitacion
    ])

    # 3. Ejecutar la función principal del pipeline
    df_final = obtener_dataframe()

    # 4. Verificar el resultado final
    assert not df_final.empty
    assert isinstance(df_final, pd.DataFrame)
    assert 'estacion' in df_final.columns
    assert df_final.loc[0, 'estacion'] == 'Otoño'
    # Verificar que las columnas clave existen después de todo el proceso
    assert 'PHT (µg/l)' in df_final.columns
    assert 'PRS (µg/l)' in df_final.columns
    assert 'Dominancia de Cianobacterias (%)' in df_final.columns

# Marca esta prueba como 'db' porque necesita la base de datos de Docker
@pytest.mark.db
@pytest.mark.db
def test_database_listener_triggers_update(mocker):
    """
    Prueba que una notificación de la base de datos dispara la actualización.
    """
    # 1. Mockear la función final que queremos verificar que se llama.
    mock_actualizar_df = mocker.patch('backend.actualizar_df')

    # 2. Configurar el hilo del listener y los parámetros de la BD de prueba
    stop_event = threading.Event()
    test_db_params = {
        'dbname': 'testdb',
        'user': 'testuser',
        'password': 'testpass',
        'host': 'localhost',
        'port': '54321'
    }

    # Creamos el hilo, pasándole los parámetros de la BD de prueba
    listener_thread = threading.Thread(
        target=database_listener,
        # Usamos kwargs para pasar los argumentos por nombre, es más claro
        kwargs={'stop_event': stop_event, 'test_db_params': test_db_params},
        daemon=True
    )

    try:
        # 3. Iniciar el listener en segundo plano
        listener_thread.start()
        time.sleep(2) # Esperar para asegurar que el listener se conecte

        # 4. Conectarse a la MISMA BD de prueba y enviar la notificación
        conn_string = f"dbname='{test_db_params['dbname']}' user='{test_db_params['user']}' password='{test_db_params['password']}' host='{test_db_params['host']}' port={test_db_params['port']}"
        with psycopg2.connect(conn_string) as conn:
            conn.autocommit = True
            with conn.cursor() as curs:
                curs.execute("NOTIFY datos_agua_actualizados, 'test_payload';")
        
        # 5. Esperar a que el listener procese la notificación
        # Puede ser útil aumentar un poco el tiempo si la prueba es inestable
        time.sleep(2) 
        
        # 6. La aserción clave: ahora debería pasar
        mock_actualizar_df.assert_called_once()

    finally:
        # 7. Limpieza
        stop_event.set()
        if listener_thread.is_alive():
            listener_thread.join(timeout=2)



def test_actualizar_df_llama_a_reentrenar_modelos(mocker):
    """Verifica que el pipeline de actualización dispara el re-entrenamiento."""
    # Simular funciones para que la prueba se enfoque solo en el subprocess
    mocker.patch('backend.obtener_dataframe', return_value=pd.DataFrame({'a': [1]})) # Devuelve un DF no vacío
    mocker.patch('pandas.DataFrame.to_sql') # SIMULAR to_sql DIRECTAMENTE
    mocker.patch('backend.recargar_modelos') # Evita que se intenten cargar archivos

    # El mock clave: simular subprocess.run
    mock_subprocess_run = mocker.patch('subprocess.run')

    # Ejecutar la función que debería llamar al subproceso
    actualizar_df()

    # Verificar que subprocess.run fue llamado
    mock_subprocess_run.assert_called_once()
    
    # Opcional: Verificar que fue llamado con los argumentos correctos
    ruta_script_esperada = os.path.join(os.path.dirname(os.path.abspath('backend.py')), 'entrenar_modelos.py')
    args_llamada = mock_subprocess_run.call_args[0][0] # Extrae los argumentos de la llamada
    assert args_llamada == ['python3', ruta_script_esperada]


# `mocker` viene de pytest-mock
def test_get_options_endpoint(client, mocker):
    """Prueba que el endpoint /get-options funcione correctamente."""
    # 1. Simular (mock) pd.read_sql para que no dependa de la BD real
    mock_df = pd.DataFrame({'codigo_perfil': ['C1', 'TAC1', 'C1', np.nan]})
    mocker.patch('pandas.read_sql', return_value=mock_df)

    # 2. Llamar al endpoint usando el cliente de prueba
    response = client.get('/get-options')

    # 3. Verificar los resultados
    assert response.status_code == 200
    # El endpoint debe devolver una lista JSON con valores únicos y sin nulos
    expected_data = ['C1', 'TAC1']
    assert sorted(response.json) == sorted(expected_data)

def test_predict_endpoint_sitio_unico(client, mocker):
    """Prueba el endpoint /predict para un sitio específico."""
    # 1. Simular la función `hacer_prediccion_para_sitio` para no depender
    #    de todo el pipeline de ML (modelos, db, etc.)
    #    Esto es una prueba de integración del *endpoint*, no del modelo.
    mock_prediction_result = {
        'codigo_perfil': 'C1',
        'fecha_prediccion': '2025-07-28T00:00:00',
        'Clorofila': {'prediccion': 'Alerta'},
        'Cianobacterias': {'prediccion': 'Vigilancia'},
        'Dominancia': {'prediccion': 'No Dominante'}
    }
    mocker.patch('backend.hacer_prediccion_para_sitio', return_value=mock_prediction_result)

    # 2. Hacer la petición POST al endpoint
    response = client.post('/predict', json={'option': 'C1'})

    # 3. Verificar la respuesta
    assert response.status_code == 200
    # La API devuelve una lista, incluso para una sola predicción
    assert isinstance(response.json, list)
    assert len(response.json) == 1
    assert response.json[0]['codigo_perfil'] == 'C1'
    assert response.json[0]['Clorofila']['prediccion'] == 'Alerta'

def test_predict_endpoint_sin_sitio(client):
    """Prueba que el endpoint /predict devuelva un error 400 si no se envía la opción."""
    response = client.post('/predict', json={}) # Enviamos un JSON vacío
    assert response.status_code == 400
    assert 'error' in response.json
    assert response.json['error'] == 'No se especificó un sitio.'