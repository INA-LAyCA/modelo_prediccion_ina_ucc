# tests/entrenamiento/test_entrenar_modelos.py
import pytest
import pandas as pd
import numpy as np

# Importamos las funciones que queremos probar desde el script original
from entrenar_modelos import (
    classify_chlorophyll_alerta,
    classify_cyanobacteria_alerta,
    classify_dominance_alerta,
    preprocess_and_feature_engineer
)

# --- Pruebas para las funciones de clasificación ---

def test_classify_chlorophyll_alerta():
    """Prueba la lógica de clasificación de clorofila."""
    assert classify_chlorophyll_alerta(5) == 0      # Nivel Vigilancia
    assert classify_chlorophyll_alerta(10) == 1     # Nivel Alerta (límite)
    assert classify_chlorophyll_alerta(20) == 1     # Nivel Alerta
    assert classify_chlorophyll_alerta(25) == 2     # Nivel Emergencia
    assert pd.isna(classify_chlorophyll_alerta(np.nan)) # Manejo de nulos

def test_classify_cyanobacteria_alerta():
    """Prueba la lógica de clasificación de cianobacterias."""
    # Los valores de entrada están en cel/L, la función convierte a cel/mL
    assert classify_cyanobacteria_alerta(4000 * 1000) == 0  # Vigilancia
    assert classify_cyanobacteria_alerta(5000 * 1000) == 1  # Alerta (límite)
    assert classify_cyanobacteria_alerta(50000 * 1000) == 1 # Alerta
    assert classify_cyanobacteria_alerta(70000 * 1000) == 2 # Emergencia
    assert pd.isna(classify_cyanobacteria_alerta(np.nan)) # Nulos

def test_classify_dominance_alerta():
    """Prueba la lógica de clasificación de dominancia."""
    assert classify_dominance_alerta(49) == 0       # No Dominante
    assert classify_dominance_alerta(50) == 1       # Dominante (límite)
    assert classify_dominance_alerta(80) == 1       # Dominante
    assert pd.isna(classify_dominance_alerta(np.nan)) # Nulos

# --- Prueba de Integración para una función más compleja ---

def test_preprocess_and_feature_engineer_crea_columnas_lag():
    """
    Prueba que la función de preprocesamiento cree correctamente
    las columnas de lag y rolling window.
    """
    # 1. Crear un DataFrame de entrada de ejemplo
    data = {
        'fecha': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'codigo_perfil': ['C1', 'C1', 'C1'],
        'T° (°C)': [15, 16, 17] # Una de las features usadas para lags
    }
    df_raw = pd.DataFrame(data)

    # 2. Ejecutar la función
    df_processed = preprocess_and_feature_engineer(df_raw)

    # 3. Verificar los resultados
    # La columna de lag 1 para el tercer registro debe ser el valor del segundo
    expected_lag1_value = 16
    assert 'T° (°C)lag1' in df_processed.columns
    assert df_processed.loc[2, 'T° (°C)lag1'] == expected_lag1_value

    # La columna de media móvil de ventana 3 para el tercer registro debe ser la media de (15, 16, 17)
    expected_roll_mean3_value = np.mean([15, 16, 17])
    assert 'T° (°C)roll_mean3' in df_processed.columns
    assert df_processed.loc[2, 'T° (°C)roll_mean3'] == expected_roll_mean3_value