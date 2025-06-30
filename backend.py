import pandas as pd
import numpy as np
import select
import os
import joblib
import subprocess
import threading
import time
import psycopg2
import logging
import tensorflow as tf
from sqlalchemy import create_engine
from sqlalchemy import text
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from entrenar_modelos import preprocess_and_feature_engineer

# Configuración de Flask
app = Flask(__name__)
CORS(app)

# Conexión a la base de datos water quality
usuario = 'postgres'
contraseña = 'postgres'
host = '192.168.191.230'
puerto = '5434'
nombre_base_datos = 'water_quality'

# Conexión a la base de datos alerts, usuario y contraseña es la misma
host2 = '192.168.191.164'
puerto2 = '5433'
nombre_base_datos2 = 'alerts'
nombre_base_modelo = 'model_data'

# Usando SQLAlchemy para crear la conexión
engine = create_engine(f'postgresql+psycopg2://{usuario}:{contraseña}@{host}:{puerto}/{nombre_base_datos}')
engine2 = create_engine(f'postgresql+psycopg2://{usuario}:{contraseña}@{host2}:{puerto2}/{nombre_base_datos2}')
engine3 = create_engine(f'postgresql+psycopg2://{usuario}:{contraseña}@{host2}:{puerto2}/{nombre_base_modelo}')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

APP_STATUS = {'is_retraining': False}
status_lock = threading.Lock()

# Función para obtener y procesar el DataFrame
def obtener_dataframe():
    try:
        df = pd.read_sql("SELECT * from vistaconjunto", engine)
        dfP = pd.read_sql("SELECT * from vista_alerts", engine2)
    except Exception as e:
        print(f"Error leyendo de la base de datos: {e}")
        return pd.DataFrame()
    
    base_columns = ['id_registro', 'condicion_termica', 'fecha', 'codigo_perfil', 'descripcion_estratificacion']
    df_base = df[base_columns].drop_duplicates()
    
    df_pivot = df.pivot_table(index='id_registro', columns='parametro', values='valor_parametro', aggfunc='first').reset_index()
    df_final = pd.merge(df_base, df_pivot, on='id_registro', how='left')
    
    cols_cianobact = ['Anabaena', 'Anabaenopsis', 'Aphanizomenon', 'Aphanocapsa', 'Aphanothece', 'Geitlerinema', 'Merismopedia', 'Chroococcus', 'Nostoc', 'Microcystis', 'Oscillatoria', 'Phormidium', 'Planktothrix', 'Pseudoanabaena', 'Raphydiopsis', 'Romeria', 'Spirulina', 'Dolichospermum', 'Leptolyngbya', 'Synechococcus']
    
    # Lógica robusta para cianobacterias
    cols_cianobact_presentes = [col for col in cols_cianobact if col in df_final.columns]
    if cols_cianobact_presentes:
        df_final['Cianobacterias Total'] = df_final[cols_cianobact_presentes].apply(lambda row: np.nan if row.isnull().any() else row.sum(), axis=1)
        df_final.drop(columns=cols_cianobact_presentes, inplace=True)
    else:
        df_final['Cianobacterias Total'] = np.nan
    
    df_final['fecha'] = pd.to_datetime(df_final['fecha'], format='%Y-%m-%d', errors='coerce')
    df_final.dropna(subset=['fecha'], inplace=True)
    df_final = df_final[df_final['fecha'] >= pd.Timestamp('1999-07-24')]

    #Imputación Cota
    df_final = imputar_cota_m(df_final)

    #Selección de mejores fechas y eliminacion de duplicados
    df_final = seleccionar_medicion_mensual(df_final)
    
    #Imputar condicion termica
    df_final=condicion_termica(df_final, engine)

    #Union temperatura del aire
    df_final=union_temperatura_aire (df_final, engine2)
    
    df_final['fecha'] = pd.to_datetime(df_final['fecha'], errors='coerce') # Re-asegurar por si acaso

    if 'mes' not in df_final.columns: # Crear 'mes' si no lo hizo union_temperatura_aire
        df_final['mes'] = df_final['fecha'].dt.month
    
    if 'estacion' not in df_final.columns: # Crear 'estacion' si no lo hizo alguna función previa de forma consistente
        df_final['estacion'] = df_final['mes'].apply(asignar_estacion) # Usar la función helper consistente
    
    #Imputación temperatura del agua
    df_final=imputacion_temperatura_agua(df_final)

    # Eliminar registros donde ambas columnas son nulas
    df_final = df_final[~(df_final['Total Algas Sumatoria (Cel/L)'].isnull() & df_final['Cianobacterias Total'].isnull())]

    #Imputacion temperatura del aire max y min 
    df_final=imputacion_temperatura_aire(df_final)

    # --- Ejecución principal ---
    df_final, _ = imputacion_clorofila(df_final)
    df_final, _ = imputar_pht(df_final)
    df_final, _ = imputar_prs(df_final)
    df_final, resultado_imputaciones, resultados_nitrogeno = imputar_nitrogeno(df_final)
    # Cálculo de Nitrogeno Inorganico Total
    # Esta suma usa 'N-NH4 (µg/l)', 'N-NO2 (µg/l)' y 'N-NO3 (mg/l)'
    df_final['Nitrogeno Inorganico Total (µg/l)'] = df_final.apply(
        lambda row: np.nan if pd.isnull(row['N-NH4 (µg/l)']) or pd.isnull(row['N-NO2 (µg/l)']) or pd.isnull(row['N-NO3 (mg/l)']) 
        else row['N-NH4 (µg/l)'] + row['N-NO2 (µg/l)'] + (row['N-NO3 (mg/l)'] * 1000),
        axis=1
    )
    
    #Eliminar las columnas de nitrógeno individuales (incluyendo N-NO3 (mg/l)) ---
    # CÓDIGO NUEVO Y ROBUSTO
    columnas_nitrogeno_a_eliminar = [
    'N-NH4 (µg/l)',
    'N-NO2 (µg/l)',
    'N-NO3 (µg/l)',
    'N-NO3 (mg/l)'
    ]

# Filtra la lista para obtener solo las columnas que SÍ existen en el DataFrame
    columnas_existentes_para_eliminar = [
        col for col in columnas_nitrogeno_a_eliminar if col in df_final.columns
    ]

# Elimina únicamente las columnas que existen
    if columnas_existentes_para_eliminar:
        df_final.drop(columns=columnas_existentes_para_eliminar, inplace=True)
    if 'Cianobacterias Total' in df_final.columns and 'Total Algas Sumatoria (Cel/L)' in df_final.columns:
        df_final['Dominancia de Cianobacterias (%)'] = (df_final['Cianobacterias Total'] * 100) / df_final['Total Algas Sumatoria (Cel/L)']
    else:
        df_final['Dominancia de Cianobacterias (%)'] = np.nandf_final.loc[
    
    df_final['codigo_perfil'].isin(['C1', 'TAC1', 'TAC4']) & df_final['condicion_termica'].isna(),
    'condicion_termica'
    ] = 'SD'
        
    df_final=union_precipitacion(df_final, engine2)
    
    return df_final

# --- Funciones Auxiliares de Procesamiento ---

#Función union de datos de precipitación al dataframe final
def union_precipitacion (df_final, engine2):
    try:
        query4 = "SELECT * FROM vista_precipitacion_acumulada_3d;"
        df_precipitacion = pd.read_sql(query4, engine2)
        if df_precipitacion.empty or 'fecha_dia' not in df_precipitacion.columns:
            return df_final

        # sensores de interés
        sensores_interes = {
            600: '600_Bo_El_Canal', # <-- Normalizado
            700: '700_Confluencia_El_Cajon', # <-- NORMALIZADO (sin acento)
            1100: '1100_CIRSA_Villa_Carlos_Paz', # <-- Normalizado
        }

        # Reemplazar id_sensor por el nombre descriptivo
        df_precipitacion['sensor_nombre'] = df_precipitacion['sensor_id'].map(sensores_interes)
        print("Llego hasta leer los sensores")
        # Pivotear: fecha como índice, cada sensor como una columna
        df_precipitacion_pivot = df_precipitacion.pivot(
            index='fecha_dia',
            columns='sensor_id',
            values='precipitacion_acumulada_3d'
        ).reset_index()
        print("Llego hasta pivotear sensores como columnas")
        # Asegurarse que la fecha del df_final también sea tipo date
        df_final['fecha'] = pd.to_datetime(df_final['fecha']).dt.date
        df_precipitacion_pivot['fecha_dia'] = pd.to_datetime(df_precipitacion_pivot['fecha_dia']).dt.date

        # merge por fecha
        df_merged = df_final.merge(df_precipitacion_pivot, how='left', left_on='fecha', right_on='fecha_dia')
        if 'fecha_dia' in df_merged.columns:
            df_merged.drop(columns=['fecha_dia'], inplace=True)
        return df_merged
    except Exception:
        return df_final


def asignar_estacion(mes):
    if mes in [12, 1, 2]:
        return 'Verano'
    elif mes in [3, 4, 5]:
        return 'Otoño'
    elif mes in [6, 7, 8]:
        return 'Invierno'
    else:
        return 'Primavera'

def imputacion_clorofila(df_final):
    df = df_final.copy()
    if 'Clorofila (µg/l)' not in df.columns: return df, pd.DataFrame()
    
    resultados = []
    for (sitio, estacion), grupo in df.groupby(['codigo_perfil', 'estacion']):
        base_vars = ['Total Algas Sumatoria (Cel/L)', 'Cianobacterias Total']
        predictores_fosforo = ['PHT (µg/l)', 'PRS (µg/l)']
        
        if all(col in grupo.columns for col in predictores_fosforo) and grupo[predictores_fosforo].notna().all().all():
            predictores = base_vars + predictores_fosforo
        else:
            predictores = base_vars + ['T° (°C)']
        
        predictores_existentes = [p for p in predictores if p in grupo.columns]
        if not predictores_existentes: continue
            
        grupo_completo = grupo.dropna(subset=predictores_existentes + ['Clorofila (µg/l)'])
        if len(grupo_completo) < 5: continue

        X = grupo_completo[predictores_existentes]
        y = grupo_completo['Clorofila (µg/l)']
        
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        mae_modelo = -cross_val_score(modelo, X, y, scoring='neg_mean_absolute_error', cv=3).mean()
        mediana_val = y.median()
        mae_mediana = mean_absolute_error(y, [mediana_val] * len(y))
        usar_modelo = mae_modelo < mae_mediana

        if usar_modelo: modelo.fit(X, y)

        grupo_faltantes = grupo[grupo['Clorofila (µg/l)'].isna()]
        grupo_pred = grupo_faltantes.dropna(subset=predictores_existentes)
        
        for idx, fila in grupo_pred.iterrows():
            X_pred = fila[predictores_existentes].values.reshape(1, -1)
            imputado = modelo.predict(X_pred)[0] if usar_modelo else mediana_val
            df.loc[df['id_registro'] == fila['id_registro'], 'Clorofila (µg/l)'] = imputado
            resultados.append({'id_registro': fila['id_registro'], 'valor_imputado': imputado, 'metodo': 'modelo' if usar_modelo else 'mediana'})
            
    return df, pd.DataFrame(resultados)


def imputar_pht(df_final):
    df = df_final.copy()
    if 'PHT (µg/l)' not in df.columns: return df, pd.DataFrame()

    resultados = []
    predictores_posibles = ['Clorofila (µg/l)', 'Total Algas Sumatoria (Cel/L)', 'Cianobacterias Total', 'T° (°C)']
    
    for (sitio, estacion), grupo in df.groupby(['codigo_perfil', 'estacion']):
        predictores_existentes = [p for p in predictores_posibles if p in grupo.columns]
        if not predictores_existentes: continue

        grupo_completo = grupo.dropna(subset=predictores_existentes + ['PHT (µg/l)'])
        if len(grupo_completo) < 5: continue
        
        X = grupo_completo[predictores_existentes]
        y = grupo_completo['PHT (µg/l)']
        
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        mae_modelo = -cross_val_score(modelo, X, y, scoring='neg_mean_absolute_error', cv=3).mean()
        mediana_val = y.median()
        mae_mediana = mean_absolute_error(y, [mediana_val] * len(y))
        usar_modelo = mae_modelo < mae_mediana
        
        if usar_modelo: modelo.fit(X, y)
            
        grupo_faltantes = grupo[grupo['PHT (µg/l)'].isna()]
        grupo_pred = grupo_faltantes.dropna(subset=predictores_existentes)
        
        for idx, fila in grupo_pred.iterrows():
            X_pred = fila[predictores_existentes].values.reshape(1, -1)
            imputado = modelo.predict(X_pred)[0] if usar_modelo else mediana_val
            df.loc[df['id_registro'] == fila['id_registro'], 'PHT (µg/l)'] = imputado
            resultados.append({'id_registro': fila['id_registro'], 'valor_imputado': imputado, 'metodo': 'modelo' if usar_modelo else 'mediana'})
            
    return df, pd.DataFrame(resultados)

def imputar_prs(df_final):
    df = df_final.copy()
    if 'PRS (µg/l)' not in df.columns: return df, pd.DataFrame()

    resultados = []
    predictores_posibles = ['PHT (µg/l)', 'Clorofila (µg/l)', 'Total Algas Sumatoria (Cel/L)', 'Cianobacterias Total', 'T° (°C)']
    
    for (sitio, estacion), grupo in df.groupby(['codigo_perfil', 'estacion']):
        predictores_existentes = [p for p in predictores_posibles if p in grupo.columns]
        if not predictores_existentes: continue

        grupo_completo = grupo.dropna(subset=predictores_existentes + ['PRS (µg/l)'])
        if len(grupo_completo) < 5: continue
        
        X = grupo_completo[predictores_existentes]
        y = grupo_completo['PRS (µg/l)']
        
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        mae_modelo = -cross_val_score(modelo, X, y, scoring='neg_mean_absolute_error', cv=3).mean()
        mediana_val = y.median()
        mae_mediana = mean_absolute_error(y, [mediana_val] * len(y))
        usar_modelo = mae_modelo < mae_mediana
        
        if usar_modelo: modelo.fit(X, y)
            
        grupo_faltantes = grupo[grupo['PRS (µg/l)'].isna()]
        grupo_pred = grupo_faltantes.dropna(subset=predictores_existentes)
        
        for idx, fila in grupo_pred.iterrows():
            X_pred = fila[predictores_existentes].values.reshape(1, -1)
            imputado = modelo.predict(X_pred)[0] if usar_modelo else mediana_val
            df.loc[df['id_registro'] == fila['id_registro'], 'PRS (µg/l)'] = imputado
            resultados.append({'id_registro': fila['id_registro'], 'valor_imputado': imputado, 'metodo': 'modelo' if usar_modelo else 'mediana'})
            
    return df, pd.DataFrame(resultados)

def imputar_nitrogeno(df_final):
    df = df_final.copy()
    
    # --- INICIO DE LA LÓGICA ROBUSTA ---

    # 1. Chequeo inicial: Si la columna base no existe, no podemos hacer nada.
    if 'N-NO3 (mg/l)' not in df.columns:
        # Creamos columnas vacías para que el resto del pipeline no falle
        # si espera estas columnas más adelante.
        df['N-NH4 (µg/l)'] = np.nan
        df['N-NO2 (µg/l)'] = np.nan
        # Devolvemos el DataFrame y resultados vacíos.
        return df, pd.DataFrame(), pd.DataFrame()

    # Si la columna existe, procedemos.
    df['N-NO3 (µg/l)'] = df['N-NO3 (mg/l)'] * 1000

    # 2. Selección robusta de predictores
    objetivo = ['N-NH4 (µg/l)', 'N-NO2 (µg/l)', 'N-NO3 (µg/l)']
    variables_predictoras_posibles = [
        'PHT (µg/l)', 'PRS (µg/l)', 'Clorofila (µg/l)', 
        'Total Algas Sumatoria (Cel/L)', 'Cianobacterias Total', 'T° (°C)'
    ]
    
    # Filtramos solo los predictores que realmente existen en el DataFrame
    variables_predictoras = [p for p in variables_predictoras_posibles if p in df.columns]

    # Si no hay predictores, no podemos imputar.
    if not variables_predictoras:
        df['N-NO3 (mg/l)'] = df['N-NO3 (µg/l)'] / 1000
        return df, pd.DataFrame(), pd.DataFrame()

    # --- FIN DE LA LÓGICA ROBUSTA ---

    resultados = []
    imputaciones = []

    # Iteramos solo sobre las columnas objetivo que existen
    objetivo_existente = [v for v in objetivo if v in df.columns]

    for (sitio, est), grupo in df.groupby(['codigo_perfil', 'estacion']):
        for variable in objetivo_existente:
            grupo_entrenamiento = grupo.dropna(subset=[variable] + variables_predictoras)
            if len(grupo_entrenamiento) < 5:
                continue

            X = grupo_entrenamiento[variables_predictoras]
            y = grupo_entrenamiento[variable]

            modelo = RandomForestRegressor(random_state=42)
            mae_modelo = -cross_val_score(modelo, X, y, scoring='neg_mean_absolute_error', cv=3).mean()
            mae_mediana = np.mean(np.abs(y - y.median()))
            usar_modelo = mae_modelo < mae_mediana

            grupo_faltantes = grupo[grupo[variable].isna()]
            grupo_pred = grupo_faltantes.dropna(subset=variables_predictoras)

            if not grupo_pred.empty:
                if usar_modelo:
                    modelo.fit(X, y)
                    predicciones = modelo.predict(grupo_pred[variables_predictoras])
                    for i, idx in enumerate(grupo_pred.index):
                        valor = predicciones[i]
                        df.loc[idx, variable] = valor
                        imputaciones.append({
                            'id_registro': df.loc[idx, 'id_registro'],
                            'variable': variable,
                            'valor_imputado': valor,
                            'codigo_perfil': sitio,
                            'estacion': est,
                            'metodo': 'modelo'
                        })
                else:
                    mediana = y.median()
                    for idx in grupo_pred.index:
                        df.loc[idx, variable] = mediana
                        imputaciones.append({
                            'id_registro': df.loc[idx, 'id_registro'],
                            'variable': variable,
                            'valor_imputado': mediana,
                            'codigo_perfil': sitio,
                            'estacion': est,
                            'metodo': 'mediana'
                        })

            valores = y.dropna()
            if len(valores) >= 2:
                rango = valores.max() - valores.min()
                error_modelo_pct = (mae_modelo / rango) * 100 if rango else np.nan
                error_mediana_pct = (mae_mediana / rango) * 100 if rango else np.nan
            else:
                rango = error_modelo_pct = error_mediana_pct = np.nan

            resultados.append({
                'codigo_perfil': sitio,
                'estacion': est,
                'variable': variable,
                'mae_modelo': mae_modelo,
                'mae_mediana': mae_mediana,
                'modelo_aplicado': usar_modelo,
                'rango': rango,
                'error_modelo_pct': error_modelo_pct,
                'error_mediana_pct': error_mediana_pct
            })

    # Convertir de vuelta a mg/l si la columna existe
    if 'N-NO3 (µg/l)' in df.columns:
        df['N-NO3 (mg/l)'] = df['N-NO3 (µg/l)'] / 1000

    return df, pd.DataFrame(imputaciones), pd.DataFrame(resultados)


def imputacion_temperatura_aire(df_final):
    # Crear columna auxiliar mes (si no existe ya)
    if 'mes' not in df_final.columns:
        df_final['mes'] = df_final['fecha'].dt.month

# Imputar temperatura del aire max y min por estación y mes
    for var in ['temperatura_max', 'temperatura_min']:
        if var in df_final.columns: # Verificar si la columna existe
            df_final[var] = pd.to_numeric(df_final[var], errors='coerce')
            df_final[var] = df_final.groupby(['estacion', 'mes'])[var].transform(lambda x: x.fillna(x.mean()))
            # Opcional: Relleno global
            if df_final[var].isnull().any():
                df_final[var].fillna(df_final[var].mean(), inplace=True)
    return df_final

def union_temperatura_aire (df_final, engine2):
    query5 = "SELECT * FROM vista_temperatura;"
    df_temp = pd.read_sql(query5, engine2)

    # Verificar y convertir a datetime si es necesario
    df_final['fecha'] = pd.to_datetime(df_final['fecha'], errors='coerce')
    df_temp['fecha_dia'] = pd.to_datetime(df_temp['fecha_dia'], errors='coerce')

    # Merge entre df_final y temperatura diaria
    df_final = df_final.merge(df_temp, left_on='fecha', right_on='fecha_dia', how='left')

    #Eliminar columna redundante
    df_final.drop(columns='fecha_dia', inplace=True)
    print("Ejemplo union temperatura: ", df_final.head())

    return df_final

def imputacion_temperatura_agua(df_final):
    df_final['T° (°C)'] = pd.to_numeric(df_final['T° (°C)'], errors='coerce')
    
    # Imputar temperatura usando el promedio por estación y por sitio
    df_final['T° (°C)'] = df_final.groupby(['codigo_perfil', 'estacion'])['T° (°C)'].transform(
        lambda x: x.fillna(x.mean())
    )
    return df_final


def imputar_cota_m(df_final):
        # Agrupar por 'fecha' y rellenar los valores nulos en 'Cota (m)'
    if 'Cota (m)' in df_final.columns:
        df_final['Cota (m)'] = pd.to_numeric(df_final['Cota (m)'], errors='coerce')
        df_final['Cota (m)'] = df_final.groupby('fecha')['Cota (m)'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
        # Considera un fillna global si aún quedan NaNs después del groupby
        df_final['Cota (m)'].fillna(df_final['Cota (m)'].mean(), inplace=True)
    return df_final

def seleccionar_medicion_mensual(df_final):
    """
    Selecciona la entrada más representativa para cada sitio dentro de cada mes.
    Los criterios son: mayor número de sitios medidos ese día en el mes,
    menor porcentaje de datos faltantes, y la fecha más reciente.
    """
    df = df_final.copy() # Trabajar sobre una copia para evitar SettingWithCopyWarning

    # 1. Crear columnas auxiliares para la agrupación y ordenamiento
    df['year_month'] = df['fecha'].dt.to_period('M')
    
    if 'codigo_perfil' in df.columns: # Solo si la columna existe
        df['num_sitios_dia_mes'] = df.groupby(['year_month', 'fecha'])['codigo_perfil'].transform('nunique')
    else:
        df['num_sitios_dia_mes'] = 1 # Valor por defecto si no hay codigo_perfil

    # 2. Calcular el porcentaje de datos faltantes para cada día
    # Manera más robusta de seleccionar columnas de medición:
    # Excluir columnas identificadoras/categóricas/auxiliares conocidas
    cols_identificadoras_y_aux = [
        'id_registro', 'condicion_termica', 'fecha', 'codigo_perfil', 
        'descripcion_estratificacion', 'z', 'year_month', 'num_sitios_dia_mes' # Incluir la recién creada
    ] 
 
    columnas_de_medicion = [col for col in df.columns if col not in cols_identificadoras_y_aux and pd.api.types.is_numeric_dtype(df[col])]
    
    if columnas_de_medicion:
        df['porcentaje_faltantes_dia'] = df[columnas_de_medicion].isna().mean(axis=1)
    else:
        df['porcentaje_faltantes_dia'] = 0.0 # Si no hay columnas de medición, no hay faltantes

    # 3. Selección de la fecha final
    columnas_ordenamiento = ['year_month', 'num_sitios_dia_mes', 'porcentaje_faltantes_dia', 'fecha']
    # Asegurar que todas las columnas de ordenamiento existen
    columnas_ordenamiento_existentes = [col for col in columnas_ordenamiento if col in df.columns]

    if len(columnas_ordenamiento_existentes) == len(columnas_ordenamiento):
         data_sorted = df.sort_values(
            by=columnas_ordenamiento_existentes,
            ascending=[True, False, True, False] # Mes asc, num_sitios desc, faltantes asc, fecha desc
        )
         # Seleccionar la primera entrada de cada mes y sitio
         df_seleccionado = data_sorted.drop_duplicates(subset=['year_month', 'codigo_perfil'], keep='first')
    else:
        # Si faltan columnas para ordenar, se devuelve el df sin este paso o se loggea una advertencia
        app.logger.warning("No se pudo realizar la selección de medición mensual representativa por falta de columnas de ordenamiento.")
        df_seleccionado = df 

    # 4 Eliminar columnas auxiliares creadas dentro de esta función
    cols_aux_a_eliminar = ['year_month', 'num_sitios_dia_mes', 'porcentaje_faltantes_dia']
    df_seleccionado = df_seleccionado.drop(columns=[col for col in cols_aux_a_eliminar if col in df_seleccionado.columns], errors='ignore')
    
    return df_seleccionado

def condicion_termica(df_principal, db_engine_param):
    """
    Calcula y actualiza la columna 'condicion_termica' en el DataFrame principal,
    siguiendo la lógica del script original del usuario.
    """
    df_a_modificar = df_principal.copy() # Trabajar sobre una copia del DataFrame principal

    # Consulta para obtener los datos de condicion_termica
    query3 = "SELECT * from vista_condicion_termica"
    df_CT = None # Inicializar df_CT por si falla la carga
    try:
        # Usar el parámetro db_engine_param en lugar del 'engine' global
        df_CT = pd.read_sql(query3, db_engine_param)
        print("Primeras filas de df_CT (vista_condicion_termica):")
        print(df_CT.head())
    except Exception as e:
        print(f"Error al leer vista_condicion_termica: {e}")
        return df_a_modificar # Devolver el df principal sin modificar si hay error

    if df_CT is None or df_CT.empty:
        print("df_CT está vacío o no se pudo cargar. No se procesará condición térmica.")
        return df_a_modificar

    print("Valores nulos en df_CT antes del procesamiento de condición térmica:")
    print(df_CT.isna().sum())

    # Agrupación por familia de perfiles
    grupos_perfiles = {
        'C': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'TAC': ['TAC1', 'TAC2', 'TAC3', 'TAC4', 'TAC5'],
        'DSA': ['DSA1', 'DSA2', 'DSA3', 'DSA4', 'DSA5'],
        'DCQ': ['DCQ1', 'DCQ2', 'DCQ3', 'DCQ4', 'DCQ5']
    }

    # Funciones auxiliares (se vuelven anidadas dentro de esta función principal)
    def asignar_grupo(perfil):
    
        for grupo, perfiles in grupos_perfiles.items():
            if perfil in perfiles:
                return grupo
        return perfil

    # Asignar columna 'grupo'
    # Asegurar que 'codigo_perfil' exista en df_CT
    if 'codigo_perfil' not in df_CT.columns:
        print("Error: La columna 'codigo_perfil' no existe en df_CT.")
        return df_a_modificar
    df_CT['grupo'] = df_CT['codigo_perfil'].apply(asignar_grupo)

    # --- PASO 1: Propagar ---
    def propagar_condicion(grupo_df_prop): # Renombrado para evitar conflicto
        # Asegurar que 'condicion_termica' exista
        if 'condicion_termica' not in grupo_df_prop.columns or grupo_df_prop['condicion_termica'].notna().sum() == 0:
            return grupo_df_prop
        valor_presente = grupo_df_prop['condicion_termica'].dropna().iloc[0]
        # Crear una copia para modificar de forma segura dentro del apply
        grupo_df_mod = grupo_df_prop.copy()
        grupo_df_mod['condicion_termica'] = valor_presente
        return grupo_df_mod

    # Asegurar que 'condicion_termica', 'fecha' y 'grupo' existan en df_CT
    required_cols_prop = ['condicion_termica', 'fecha', 'grupo']
    if not all(col in df_CT.columns for col in required_cols_prop):
        missing_str = ", ".join(list(set(required_cols_prop) - set(df_CT.columns)))
        print(f"Error: Faltan columnas ({missing_str}) para propagar condición térmica en df_CT.")
        return df_a_modificar
    
    # Convertir fecha a datetime si no lo es
    df_CT['fecha'] = pd.to_datetime(df_CT['fecha'], errors='coerce')
    df_CT.dropna(subset=['fecha'], inplace=True)


    df_CT = df_CT.groupby(['fecha', 'grupo'], group_keys=False).apply(propagar_condicion).reset_index(drop=True)

    # --- PASO 2: Calcular ---
    def calcular_condicion(grupo_df_calc_input): # Renombrado para evitar conflicto
        grupo_df_calc = grupo_df_calc_input.copy() # Trabajar sobre copia

        grupo_df_calc = grupo_df_calc.sort_values('z').reset_index(drop=True)
        
        # Si ya tiene valor, no recalculamos
        if 'condicion_termica' in grupo_df_calc.columns and grupo_df_calc['condicion_termica'].notna().all():
            return grupo_df_calc
        
        # Si no hay al menos dos puntos válidos, dejar como NaN
        if grupo_df_calc['valor'].notna().sum() < 2 or grupo_df_calc['z'].notna().sum() < 2:
            grupo_df_calc['condicion_termica'] = np.nan # Asignar NaN al grupo
            return grupo_df_calc
        
        temp_dif = grupo_df_calc['valor'].diff().abs().round(1)
        prof_dif = grupo_df_calc['z'].diff().abs()


        for i in range(1, len(grupo_df_calc)):
            t = temp_dif.iloc[i]
            z = prof_dif.iloc[i]

            if pd.notna(t) and pd.notna(z):
                if z <= 1 and t >= 1.0:
                    grupo_df_calc['condicion_termica'] = 'ESTRATIFICADA'
                    return grupo_df_calc
                elif z <= 1 and t == 0.9:
                    grupo_df_calc['condicion_termica'] = 'INDETERMINACION'
                    return grupo_df_calc
                elif z > 1 and t > 1.0:
                    grupo_df_calc['condicion_termica'] = 'INDETERMINACION'
                    return grupo_df_calc

        # Si no se cumplió ninguna condición en el bucle, se considera mezcla
        grupo_df_calc['condicion_termica'] = 'MEZCLA'
        return grupo_df_calc

    # Aplicar cálculo
    # Asegurar columnas necesarias
    df_CT = df_CT.groupby(['fecha', 'grupo']).apply(calcular_condicion).reset_index(drop=True)

    # --- Parte final: unir con df_a_modificar ---
    # Creamos diccionario con las condiciones calculadas que NO son nulas
    cond_term_dict = df_CT.dropna(subset=['condicion_termica'])[['id_registro', 'condicion_termica']].drop_duplicates()
    cond_term_dict = cond_term_dict.set_index('id_registro')['condicion_termica'].to_dict()

    # Aplicar reemplazo solo donde df_final tiene NaN en 'condicion_termica'
    df_a_modificar['condicion_termica'] = df_a_modificar.apply(
        lambda row: cond_term_dict.get(row['id_registro'], row['condicion_termica']) if pd.isna(row['condicion_termica']) else row['condicion_termica'],
        axis=1
    )

    return df_a_modificar

# CARGA Y RECARGA DE MODELOS

# =======================================================================
# --- CARGA DE MODELOS Y ARTEFACTOS AL INICIO DE LA APP ---
# =======================================================================
print("Cargando modelos y artefactos entrenados...")
SITIOS_A_ANALIZAR   = ['C1','TAC1','TAC4','DSA1','DCQ1']
TARGETS_A_PREDECIR  = ['Clorofila','Cianobacterias','Dominancia']
CLASS_LABELS_MAP_ALERTA = {
    0: "Vigilancia",
    1: "Alerta",
    2: "Emergencia"
}
CLASS_LABELS_MAP_ALERTA_DOMINANCIA = {
    0: "No Dominante",
    1: "Dominante",
}

def recargar_modelos():
    """
    Vuelve a leer de disco todos los artefactos y actualiza la variable global modelos_cargados.
    """
    print("🔄 Recargando modelos y artefactos entrenados…")
    temp = { target: {} for target in TARGETS_A_PREDECIR }

    for target in TARGETS_A_PREDECIR:
        for sitio in SITIOS_A_ANALIZAR:
            pkl_path   = f"modelos_entrenados/artefactos_{sitio}_{target}.pkl"
            keras_path = f"modelos_entrenados/modelo_{sitio}_{target}.keras"

            if not os.path.exists(pkl_path):
                print(f"  ⚠️ No encontrado: {pkl_path}")
                continue

            try:
                artefactos = joblib.load(pkl_path)
                # Si hay un modelo Keras separado, lo cargamos también
                if os.path.exists(keras_path):
                    artefactos['modelo'] = tf.keras.models.load_model(keras_path)
                temp[target][sitio] = artefactos
                print(f"  • {sitio}-{target} cargado")
            except Exception as e:
                print(f"  ❌ Error cargando {sitio}-{target}: {e}")

    # Una vez construido todo el dict sin errores, actualizamos la variable global
    with modelos_lock:
        global modelos_cargados
        modelos_cargados = temp

    print("✅ Recarga de modelos completada.\n")

# LÓGICA DE ACTUALIZACIÓN

proceso_lock = threading.Lock()
proceso_status = {
    "running": False,        # Indica si ya hay un proceso en ejecución
    "message": "Inactivo"    # Mensaje de estado, útil para exponer vía /get-status
}
modelos_lock = threading.Lock()
modelos_cargados = {}

DELAY_SEGUNDOS = 10
_debounce_timer = None
_changed_tables = set()

def _schedule_update():
    """Inicia/reinicia el timer para llamar a actualizar_df() tras DELAY_SEGUNDOS."""
    global _debounce_timer
    if _debounce_timer and _debounce_timer.is_alive():
        _debounce_timer.cancel()
    _debounce_timer = threading.Timer(DELAY_SEGUNDOS, _run_update)
    _debounce_timer.daemon = True
    _debounce_timer.start()

def _run_update():
    """Se ejecuta cuando pasan DELAY_SEGUNDOS sin nuevas notificaciones."""
    global _changed_tables
    print(f"[Listener] Ejecutando actualizar_df(), cambios de tablas: {_changed_tables}")
    actualizar_df()
    _changed_tables.clear()

def database_listener(stop_event=None, test_db_params=None):
    """
    Función del listener, ahora configurable para poder usar una base de datos de prueba.
    """
    if test_db_params:
        # Si la prueba nos da parámetros, los usamos.
        print(f"LISTENER: Usando parámetros de BD de prueba: {test_db_params['dbname']}")
        conn_string = (
            f"dbname='{test_db_params['dbname']}' "
            f"user='{test_db_params['user']}' "
            f"password='{test_db_params['password']}' "
            f"host='{test_db_params['host']}' "
            f"port='{test_db_params['port']}'"
        )
    else:
        # Si no, usa los parámetros globales de producción como antes.
        print(f"LISTENER: Usando parámetros de BD de producción: {nombre_base_datos}")
        conn_string = f"dbname='{nombre_base_datos}' user='{usuario}' password='{contraseña}' host='{host}' port='{puerto}'"

    # El resto de la función se mantiene igual, pero ahora usa el conn_string correcto.
    while True:
        if stop_event and stop_event.is_set():
            print("LISTENER: Evento de parada recibido, terminando hilo.")
            break

        try:
            conn = psycopg2.connect(conn_string)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            print("LISTENER: Conectado a la base de datos y escuchando notificaciones...")
            
            curs = conn.cursor()
            curs.execute("LISTEN datos_agua_actualizados;")

            while True:
                if stop_event and stop_event.is_set():
                    break
                
                if select.select([conn], [], [], 1) == ([], [], []):
                    continue
                
                conn.poll()
                while conn.notifies:
                    notification = conn.notifies.pop(0)
                    logging.info(f"LISTENER: Notificación recibida en '{notification.channel}'")
                    actualizar_df() # Llamamos a la función de actualización

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            logging.warning(f"LISTENER: Error de conexión: {e}. Reconectando en 5 segundos...")
            time.sleep(30)
        except Exception as e:
            logging.error(f"LISTENER: Ocurrió un error inesperado: {e}. Reiniciando en 5 segundos...")
            time.sleep(30)

def actualizar_df():
    """
    Función "trabajadora". Llama al pipeline pesado y guarda el resultado.
    Se ejecuta en un hilo para no bloquear al listener o a la app.
    """
    # with app.app_context() es una buena práctica para que el hilo
    # tenga acceso al contexto de la aplicación si fuera necesario.
    
    with app.app_context():
        print("WORKER: Iniciando `generar_dataframe_procesado()`...")
        global APP_STATUS
        with proceso_lock:
            if proceso_status["running"]:
                print("WORKER: Ya hay un proceso en curso, saliendo.")
                return
            proceso_status["running"] = True

        try:
            with status_lock:
                APP_STATUS['is_retraining'] = True
            df_procesado = obtener_dataframe()

            if df_procesado is not None and not df_procesado.empty:
                nombre_tabla_destino = 'dataframe'
                print(f"WORKER: Guardando DataFrame en la tabla '{nombre_tabla_destino}'...")
                df_procesado.to_sql(nombre_tabla_destino, engine3, if_exists='replace', index=False)
                print("WORKER: ¡Guardado en la base de datos exitoso!")
                print("WORKER: Reentrenando modelos")
                reentrenar_modelos()
                print("WORKER: Recargando en memoria los modelos entrenados…")
                recargar_modelos()
                print("WORKER: Modelos recargados exitosamente.")
            else:
                print("WORKER: El procesamiento no generó un DataFrame. No se guardó nada.")
       
            print("WORKER: Recargando modelos en memoria…")
            recargar_modelos()
            print("WORKER: Modelos recargados correctamente.")
        except Exception as e:
            print(f"WORKER: ERROR en el proceso de actualización en segundo plano: {e}")
        finally:
            with status_lock:
                APP_STATUS['is_retraining'] = False
            with proceso_lock:
                proceso_status["running"] = False
                proceso_status["message"] = "Inactivo"

def reentrenar_modelos():
    """
    Lanza el script entrenar_modelos.py como un subproceso y muestra su salida.
    """
    ruta_script = os.path.join(os.path.dirname(__file__), 'entrenar_modelos.py')
    try:
        resultado = subprocess.run(
            ['python3', ruta_script],
            capture_output=True,
            text=True,
            check=True
        )
        print("WORKER: Salida de entrenar_modelos.py:")
        print(resultado.stdout)
    except subprocess.CalledProcessError as e:
        print("WORKER ERROR: El retraining falló:")
        print(e.stderr)
        # opcional: re-lanzar o marcar error según tu lógica

# PREDICCION

def hacer_prediccion_para_sitio(sitio: str) -> dict:
    """
    Versión mejorada que devuelve la predicción JUNTO con las métricas
    del modelo que la realizó.
    """
    resultado = {'codigo_perfil': sitio}

    # 1) Traer historial completo (ordenado)
    df_raw = pd.read_sql(
        "SELECT * FROM dataframe WHERE codigo_perfil = %s ORDER BY fecha",
        engine3, params=(sitio,)
    )
    if df_raw.empty:
        resultado['error'] = "No hay datos históricos"
        return resultado

    # 2) Preprocesar TODO el histórico
    df_proc = preprocess_and_feature_engineer(df_raw)

    if df_proc.empty:
        logging.warning(f"Se omitió la predicción para {sitio} porque el preprocesamiento resultó en un DataFrame vacío.")
        resultado['error'] = "Los datos disponibles no son válidos para la predicción."
        return resultado
    # 

    # 3) Tomar la fila más reciente
    ultimo = df_proc.iloc[[-1]]

    try:
        ultima_fecha_conocida = pd.to_datetime(ultimo['fecha'].iloc[0])
        fecha_prediccion = ultima_fecha_conocida + pd.offsets.DateOffset(months=1)
    except (IndexError, TypeError):
        fecha_prediccion = pd.Timestamp.now().date()
    resultado['fecha_prediccion'] = fecha_prediccion.isoformat()

    # 4) Para cada target, filtrar, escalar y predecir
    for target in TARGETS_A_PREDECIR:
        with modelos_lock:
            artef = modelos_cargados.get(target, {}).get(sitio)
        if not artef:
            resultado[target] = "Modelo no disponible"
            continue

        # Extraer scaler, lista de columnas y modelo
        scaler = artef['scaler']
        best_features = artef['best_features']
        modelo = artef['modelo']
        model_info = artef.get('model_info', {})

        # 5) Construir y escalar vector de predicción
        X_df = ultimo.reindex(columns=best_features, fill_value=0)
        X_scaled = scaler.transform(X_df)

        # 6) Predecir
        if isinstance(modelo, tf.keras.Model):
            probs = modelo.predict(X_scaled, verbose=0)
            cls = int(np.argmax(probs, axis=1)[0])
        else:
            cls = int(modelo.predict(X_scaled)[0])

        if target == 'Dominancia':
           etiqueta_predicha = CLASS_LABELS_MAP_ALERTA_DOMINANCIA.get(cls, "Desconocido")
        else:
            etiqueta_predicha = CLASS_LABELS_MAP_ALERTA.get(cls, "Desconocido")

        
        try:
            guardar_prediccion_historica(
                codigo_perfil=sitio,
                fecha_prediccion=fecha_prediccion,
                target=target,
                clase_alerta=cls,
                etiqueta_predicha=etiqueta_predicha
            )
            logging.info(f"Predicción para {sitio}/{target} guardada en la base de datos.")
        except Exception as e:
            # Si falla el guardado, solo lo informamos y continuamos. No debe detener la app.
            logging.error(f"FALLO AL GUARDAR PREDICCIÓN para {sitio}/{target}: {e}")
        

        # 7) Construir el diccionario de respuesta enriquecido
        resultado[target] = {
            'prediccion': etiqueta_predicha,
            'modelo_usado': model_info.get('modelo', 'N/D'),
            'f1_score_cv': round(model_info.get('f1_macro_cv', 0), 4),
            'roc_auc_cv': round(model_info.get('roc_auc_cv', 0), 4),
            # Convertimos los params a string por si contienen tipos no serializables a JSON
            'hiperparametros': str(model_info.get('best_params', {}))
        }

    return resultado

# GUARDAR PREDICCIONES

def guardar_prediccion_historica(codigo_perfil, fecha_prediccion, target, clase_alerta, etiqueta_predicha):
    """
    Inserta o actualiza (upsert) una fila en predicciones_historicas.
    """
    sql = text("""
    INSERT INTO predicciones_historicas
      (codigo_perfil, fecha_prediccion, target, clase_alerta, etiqueta_predicha)
    VALUES (:codigo_perfil, :fecha_prediccion, :target, :clase_alerta, :etiqueta_predicha)
    ON CONFLICT (codigo_perfil, fecha_prediccion, target)
    DO UPDATE SET
      clase_alerta        = EXCLUDED.clase_alerta,
      etiqueta_predicha        = EXCLUDED.etiqueta_predicha,
      timestamp_ejecucion = now();
    """)
    params = {
        'codigo_perfil':    str(codigo_perfil),
        'fecha_prediccion': fecha_prediccion,                      # datetime.date está bien
        'target':           str(target),
        'clase_alerta':     int(clase_alerta),                     # <- aquí el cast
        'etiqueta_predicha': str(etiqueta_predicha)
    }
    with engine3.begin() as conn:
        conn.execute(sql, params)


# ENDPOINTS:

@app.route('/status', methods=['GET'])
def get_status():
    """
    Endpoint simple para que el frontend pueda consultar el estado del reentrenamiento.
    """
    with status_lock:
        is_retraining = APP_STATUS.get('is_retraining', False)
    
    status = "retraining" if is_retraining else "idle"
    
    return jsonify({'status': status})

# Consulta a la base de datos para obtener los valores de 'codigo_perfil' para el menu desplegable     
@app.route('/get-options', methods=['GET'])
def get_options():
    try:
        query = "SELECT DISTINCT codigo_perfil FROM vistaconjunto"
        df = pd.read_sql(query, engine)
        options = df['codigo_perfil'].dropna().unique().tolist()
        return jsonify(options)
    except Exception as e:
        return jsonify({'error': str(e)})

#consulta dataframe de model_data
@app.route('/datos', methods=['GET'])
def ver_datos():
    nombre_tabla = 'dataframe'
    query = f"SELECT * FROM {nombre_tabla}"

    try:
        df = pd.read_sql(query, engine3)
        if df.empty:
            return jsonify({'message': 'No hay datos procesados disponibles.', 'data': []}), 200

        df_serializable = df.replace({np.nan: None, pd.NaT: None})
        data = df_serializable.to_dict(orient='records')
        
        return jsonify(data)

    except Exception as e:
        print(f"Error al leer la tabla '{nombre_tabla}' desde engine3: {e}")
        # Devolver una respuesta de error JSON válida
        return jsonify({'error': 'No se pudieron obtener los datos del servidor.'}), 500

#consulta tabla de predicciones_historicas de model_data
@app.route('/predicciones', methods=['GET'])
def ver_predicciones():
    nombre_tabla = 'predicciones_historicas'
    query = f"SELECT * FROM {nombre_tabla}"

    try:
        df = pd.read_sql(query, engine3)
        if df.empty:
            return jsonify({'message': 'No hay datos procesados disponibles.', 'data': []}), 200

        df_serializable = df.replace({np.nan: None, pd.NaT: None})
        data = df_serializable.to_dict(orient='records')
        
        return jsonify(data)

    except Exception as e:
        print(f"Error al leer la tabla '{nombre_tabla}' desde engine3: {e}")
        # Devolver una respuesta de error JSON válida
        return jsonify({'error': 'No se pudieron obtener los datos del servidor.'}), 500

#opcion de actualización forzada, no implementada
@app.route('/actualizar', methods=['POST'])
def ejecutar_actualizacion():
    print("Solicitud a /actualizar recibida. Iniciando hilo de procesamiento manual...")
    # Llama a la misma función 'actualizar_df' pero en un hilo separado
    thread = threading.Thread(target=actualizar_df)
    thread.start()
    # Devuelve una respuesta INMEDIATA al frontend
    return jsonify({'message': 'Proceso de actualización manual iniciado en segundo plano.'}), 202

#predicción
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    sitio = data.get('option')
    if not sitio:
        return jsonify({'error':'No se especificó un sitio.'}), 400
    sitios = SITIOS_A_ANALIZAR if sitio=='Todos' else [sitio]
    out = []
    for s in sitios:
        out.append(hacer_prediccion_para_sitio(s))
    return jsonify(out)


@app.route('/metricas-historicas', methods=['GET'])
def get_metricas_historicas():
    """
    Devuelve todo el historial de métricas de entrenamiento.
    """
    try:
        # Ordenamos por fecha para que el gráfico tenga sentido cronológico
        query = "SELECT * FROM entrenamientos_historicos ORDER BY timestamp_entrenamiento ASC"
        df = pd.read_sql(query, engine3)
        
        # Convertir NaT (Not a Time) a None para que sea compatible con JSON
        df_serializable = df.replace({pd.NaT: None})
        
        # Convertir a formato JSON
        data = df_serializable.to_dict(orient='records')
        
        return jsonify(data)
    except Exception as e:
        print(f"Error al obtener métricas históricas: {e}")
        return jsonify({'error': 'No se pudieron obtener los datos de monitorización.'}), 500


#  INICIO DE LA APLICACIÓN 
print("Iniciando el listener de la base de datos en un hilo de fondo...")
listener_thread = threading.Thread(target=database_listener, daemon=True)
listener_thread.start()

if __name__ == '__main__':
    modelos_dir = "modelos_entrenados"
    if not os.path.isdir(modelos_dir) or not os.listdir(modelos_dir):
        print(f"INFO: La carpeta '{modelos_dir}' no existe o está vacía. Iniciando primer entrenamiento automático...")
        try:
            # Llamamos a la misma función que ya usas para reentrenar
            reentrenar_modelos()
            # Después del primer entrenamiento, es crucial cargar los modelos en memoria
            recargar_modelos()
        except Exception as e:
            print(f"ERROR: Falló el primer entrenamiento automático: {e}")
    else:
        print(f"INFO: Carpeta '{modelos_dir}' encontrada. Cargando modelos existentes...")
        # Si la carpeta existe y tiene contenido, simplemente cargamos los modelos
        recargar_modelos()
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
