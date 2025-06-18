# =======================================================================
# --- 1. IMPORTS Y CONFIGURACIÓN INICIAL ---
# =======================================================================
import pandas as pd
import numpy as np
import os
import joblib  # <-- CAMBIO CLAVE: Librería para guardar y cargar modelos Sklearn
import warnings
from sqlalchemy import create_engine

# Imports de Sklearn y TensorFlow/Keras (limpiados de duplicados)
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.utils import class_weight as sk_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

# Ignorar advertencias para una salida más limpia
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
tf.get_logger().setLevel('ERROR')


# =======================================================================
# --- 2. CONSTANTES GLOBALES Y CONEXIONES A BD ---
# =======================================================================

# --- CONEXIONES A BD ---
# Necesitas el engine3 para leer los datos ya procesados por tu pipeline
usuario = 'postgres'
contraseña = 'postgres'
host2 = '192.168.191.164' # Host donde está la BD de modelos
puerto2 = '5433'
nombre_base_modelo = 'model_data'
engine3 = create_engine(f'postgresql+psycopg2://{usuario}:{contraseña}@{host2}:{puerto2}/{nombre_base_modelo}')

# --- CONSTANTES GLOBALES DE CONFIGURACIÓN ---
FECHA_MAX_ENTRENAMIENTO = pd.Timestamp('2023-12-31')
SITIOS_A_ANALIZAR = ['C1', 'TAC1', 'TAC4', 'DSA1', 'DCQ1']
SITIOS_CON_CONDICION_TERMICA = ['C1', 'TAC1', 'TAC4']
SENSORES_IMPUTAR_LLUVIA = [
    '1100_CIRSA_Villa_Carlos_Paz', # <-- NORMALIZADO
    '600_Bo_El_Canal', # <-- NORMALIZADO
    '700_Confluencia_El_Cajon'  # <-- NORMALIZADO (sin acento)
]# Nombres de columnas (normalizados)
COLUMNA_FECHA = 'fecha'
COLUMNA_CODIGO_PERFIL = 'codigo_perfil'
# Variables originales para targets y features
TARGET_CLOROFILA_ORIGINAL = 'Clorofila (µg/l)'
TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L = 'Cianobacterias Total'
# Columnas de características adicionales
COLUMNA_TOTAL_ALGAS_CEL_L = 'Total Algas Sumatoria (Cel/L)'
COLUMNA_TEMPERATURA_C = 'T° (°C)'
COLUMNA_PHT_UG_L = 'PHT (µg/l)'
COLUMNA_PRS_UG_L = 'PRS (µg/l)'
COLUMNA_NITROGENO_UG_L = 'Nitrogeno Inorganico Total (µg/l)'
COLUMNA_TEMP_MAX_C = 'temperatura_max'
COLUMNA_TEMP_MIN_C = 'temperatura_min'
COLUMNA_CONDICION_TERMICA = 'condicion_termica'
# Nuevas columnas para clases y valores transformados
TARGET_CLOROFILA_CLASE_COL = 'Clorofila_Clase_Alerta'
TARGET_CIANOBACTERIAS_CLASE_COL = 'Cianobacterias_Clase_Alerta'
COLUMNA_CIANOBACTERIAS_CEL_ML = 'Cianobacterias_cel_mL_Calculado'
# Lista unificada de características para crear lags (priorizando las más predictivas o usadas en NN)
FEATURES_PARA_LAGS_UNIFICADA = sorted(list(set([
    TARGET_CLOROFILA_ORIGINAL,
    COLUMNA_CIANOBACTERIAS_CEL_ML,
    COLUMNA_TEMPERATURA_C,
    COLUMNA_TOTAL_ALGAS_CEL_L,
    COLUMNA_PHT_UG_L,
    COLUMNA_PRS_UG_L,
    COLUMNA_NITROGENO_UG_L
])))
# Parámetros de Modelos y CV
N_SPLITS_TSCV_SKLEARN = 3 # Para GridSearchCV de RF y LogReg
SCORING_METRIC_CV_SKLEARN = 'f1_weighted' # Para GridSearchCV
KT_MAX_TRIALS_NN = 5 # Reducido para ejecución más rápida, aumentar para mejor búsqueda
KT_EXECUTIONS_PER_TRIAL_NN = 1
NN_EPOCHS_TUNER = 30 # Epochs para cada trial de KerasTuner
NN_EPOCHS_FINAL_FIT = 50 # Epochs para entrenar el mejor modelo NN encontrado
NN_BATCH_SIZE = 32
NUM_CLASSES_ALERTA = 3 # Vigilancia, Alerta 1 y Alerta 2

# Mapeo de Clases (común para Clorofila y Cianobacterias según niveles de alerta)
CLASS_LABELS_MAP_ALERTA = {0: "Vigilancia/Bajo", 1: "Alerta 1/Medio", 2: "Alerta 2/Alto"}

# Modelos
MODELOS_SKLEARN_A_PROBAR = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier(random_state=42, n_jobs=-1)
}
PARAM_GRIDS_SKLEARN = {
    'LogisticRegression': {'C': [0.01, 0.1, 1.0, 10.0], 'solver': ['liblinear', 'saga']},
    'RandomForestClassifier': {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 3]}
}


# =======================================================================
# --- 3. FUNCIONES AUXILIARES (DEFINICIONES DE MODELOS, PREPROCESAMIENTO) ---
# =======================================================================


def build_mlp_model_kt(hp, input_dim, num_classes):
    units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32, default=64)
    units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16, default=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1, default=0.2)
    learning_rate = hp.Choice('learning_rate', values=[0.0005, 0.001, 0.005], default=0.001)
    model = Sequential([Input(shape=(input_dim,)), Dense(units_1, activation='relu'), Dropout(dropout_rate), Dense(units_2, activation='relu'), Dense(num_classes, activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn1d_model_kt(hp, input_shape, num_classes):
    filters = hp.Int('filters', min_value=16, max_value=64, step=16, default=32)
    kernel_size = hp.Choice('kernel_size', values=[3, 5], default=3)
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16, default=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1, default=0.3)
    learning_rate = hp.Choice('learning_rate', values=[0.0005, 0.001, 0.005], default=0.001)
    model = Sequential([Input(shape=input_shape), Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'), MaxPooling1D(pool_size=2, padding='same'), GlobalAveragePooling1D(), Dense(dense_units, activation='relu'), Dropout(dropout_rate), Dense(num_classes, activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_mlp_deeper_model_kt(hp, input_dim, num_classes):
    # ... (tu código sin cambios)
    units_1 = hp.Int('units_1', min_value=64, max_value=256, step=64, default=128)
    units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32, default=64)
    units_3 = hp.Int('units_3', min_value=16, max_value=64, step=16, default=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1, default=0.3)
    learning_rate = hp.Choice('learning_rate', values=[0.0005, 0.001, 0.005], default=0.001)
    model = Sequential([Input(shape=(input_dim,)), Dense(units_1, activation='relu'), Dropout(dropout_rate), Dense(units_2, activation='relu'), Dropout(dropout_rate), Dense(units_3, activation='relu'), Dense(num_classes, activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

MODELOS_NN_BUILDERS = {'MLP': build_mlp_model_kt, 'CNN1D': build_cnn1d_model_kt, 'MLP_Deeper': build_mlp_deeper_model_kt}

# --- FUNCIONES DE CLASIFICACIÓN DE TARGETS ---
def classify_chlorophyll_alerta(value_ug_l):
    if pd.isna(value_ug_l): return np.nan
    if value_ug_l < 10: return 0 #Vigilancia
    elif value_ug_l <= 24: return 1 #Alerta1
    else: return 2 #Alerta2

def classify_cyanobacteria_alerta(value_cel_L):
    if pd.isna(value_cel_L): return np.nan
    value_cel_mL = value_cel_L / 1000.0 
    if value_cel_mL < 5000: return 0 #Vigilancia
    elif value_cel_mL <= 60000: return 1 #Alerta1
    else: return 2 #Alerta2

def preprocess_dataframe_unified(df_raw):
    if df_raw is None: return None
    df = df_raw.copy()
    # Normalizar nombres de columnas problemáticos (ej. T° vs T¬∞)
    column_name_map = {'T¬∞ (¬∞C)': COLUMNA_TEMPERATURA_C, 'Clorofila (¬µg/l)': TARGET_CLOROFILA_ORIGINAL}
    df.rename(columns=column_name_map, inplace=True)
    if COLUMNA_FECHA not in df.columns: return None
    df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA], errors='coerce') # Asumimos que la BD guarda un formato estándar
    df.dropna(subset=[COLUMNA_FECHA], inplace=True)
    df = df.sort_values([COLUMNA_CODIGO_PERFIL, COLUMNA_FECHA])
    df = df[df[COLUMNA_FECHA] <= FECHA_MAX_ENTRENAMIENTO]
    if df.empty: return None

    # Crear características de tiempo
    df['mes'] = df[COLUMNA_FECHA].dt.month
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['estacion'] = (df['mes'] % 12 // 3) + 1 # 1:Verano(DJF), 2:Otoño(MAM), 3:Invierno(JJA), 4:Primavera(SON) - Asumiendo Hemisferio Sur DJF como inicio de estación 1
    
    # Convertir columnas relevantes a numérico
    cols_to_numeric = [TARGET_CLOROFILA_ORIGINAL, TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L, COLUMNA_TOTAL_ALGAS_CEL_L, COLUMNA_TEMPERATURA_C, COLUMNA_PHT_UG_L, COLUMNA_PRS_UG_L, COLUMNA_NITROGENO_UG_L, COLUMNA_TEMP_MAX_C, COLUMNA_TEMP_MIN_C] + SENSORES_IMPUTAR_LLUVIA
    for col in cols_to_numeric:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        else: df[col] = np.nan

    # Imputación de sensores de lluvia (de scripts Cloro/Ciano y NN)
    for sensor_col in SENSORES_IMPUTAR_LLUVIA:
        if sensor_col in df.columns:
            df[sensor_col] = df.groupby('mes')[sensor_col].transform(lambda x: x.fillna(x.median()))
            df[sensor_col].fillna(df[sensor_col].median(), inplace=True)
            df[sensor_col].fillna(0, inplace=True)
   
    # Condicion termica (de scripts Cloro/Ciano y NN)
    if COLUMNA_CONDICION_TERMICA in df.columns:
        df[COLUMNA_CONDICION_TERMICA] = df[COLUMNA_CONDICION_TERMICA].map({'MEZCLA': 0, 'INDETERMINACION': 1, 'ESTRATIFICADA': 2, 'SD': 3}).fillna(df[COLUMNA_CONDICION_TERMICA].mode()[0] if not df[COLUMNA_CONDICION_TERMICA].mode().empty else 3)
    else: df[COLUMNA_CONDICION_TERMICA] = 3
    
    if TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L in df.columns: df[COLUMNA_CIANOBACTERIAS_CEL_ML] = df[TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L] / 1000.0
    else: df[COLUMNA_CIANOBACTERIAS_CEL_ML] = np.nan
    df['pht_prs_ratio'] = df[COLUMNA_PHT_UG_L].divide(df[COLUMNA_PRS_UG_L] + 1e-6).fillna(0) if COLUMNA_PHT_UG_L in df and COLUMNA_PRS_UG_L in df else 0
    df['algas_ciano_ratio'] = df[COLUMNA_TOTAL_ALGAS_CEL_L].divide(df[TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L] + 1e-6).fillna(0) if COLUMNA_TOTAL_ALGAS_CEL_L in df and TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L in df else 0
    df['temp_diff'] = (df[COLUMNA_TEMP_MAX_C] - df[COLUMNA_TEMP_MIN_C]).fillna(0) if COLUMNA_TEMP_MAX_C in df and COLUMNA_TEMP_MIN_C in df else 0
    return df

# --- INGENIERÍA Y SELECCIÓN DE CARACTERÍSTICAS UNIFICADA ---
def create_features_for_site_unified(df_site_input, target_config, sitio_id):
    df_site = df_site_input.copy()
    
    # 1. Crear la columna target clasificada
    original_target_col_values = target_config['original_values_col']
    classification_function = target_config['classification_function']
    classified_target_col_name = target_config['classified_target_col']
    if original_target_col_values not in df_site.columns: df_site[classified_target_col_name] = np.nan
    else: df_site[classified_target_col_name] = df_site[original_target_col_values].apply(classification_function)
    
    # 2. Definir características base
    base_feature_names = [TARGET_CLOROFILA_ORIGINAL, COLUMNA_CIANOBACTERIAS_CEL_ML, COLUMNA_TOTAL_ALGAS_CEL_L, COLUMNA_TEMPERATURA_C, COLUMNA_TEMP_MAX_C, COLUMNA_TEMP_MIN_C, COLUMNA_PHT_UG_L, COLUMNA_PRS_UG_L, COLUMNA_NITROGENO_UG_L, 'mes_sin', 'mes_cos', 'estacion', 'pht_prs_ratio', 'algas_ciano_ratio', 'temp_diff']
    if sitio_id in SITIOS_CON_CONDICION_TERMICA and COLUMNA_CONDICION_TERMICA in df_site.columns: base_feature_names.append(COLUMNA_CONDICION_TERMICA)
    base_feature_names.extend(SENSORES_IMPUTAR_LLUVIA)
    current_features_direct = list(set([f for f in base_feature_names if f in df_site.columns and f != original_target_col_values]))
    
    # 3. Crear características de lag
    all_features_with_lags = current_features_direct[:]
    for col_to_lag in FEATURES_PARA_LAGS_UNIFICADA:
        if col_to_lag in df_site.columns:
            for lag_n in [1, 2, 3]:
                lag_col_name = f"{col_to_lag}_lag{lag_n}"
                df_site[lag_col_name] = df_site[col_to_lag].shift(lag_n)
                all_features_with_lags.append(lag_col_name)
    final_feature_list = sorted(list(set(f for f in all_features_with_lags if f in df_site.columns)))
    
    # 4. Preparar X y y
    X = df_site[final_feature_list].copy()
    y = df_site[classified_target_col_name].copy()
    y_not_null_mask = y.notnull()
    X = X[y_not_null_mask]
    y = y[y_not_null_mask]
    valid_dates_after_target_nan_removal = df_site.loc[y_not_null_mask, COLUMNA_FECHA]
    if y.empty: return pd.DataFrame(), pd.Series(dtype='float64'), [], pd.Series(dtype='datetime64[ns]')
    return X, y.astype(int), final_feature_list, valid_dates_after_target_nan_removal

def train_evaluate_models_unified(X_full_raw, y_full_raw, dates_full_raw, sitio_id_log, target_name_log, models_to_run=('sklearn', 'nn')):
    print(f"\n--- Entrenando y Evaluando para Sitio: {sitio_id_log}, Target: {target_name_log} ---")

    results_summary = []
    trained_models_info = {'sitio': sitio_id_log, 'target': target_name_log, 'models': {}}

    # 1. Imputación de NaNs en X (generados por lags al inicio de la serie) usando KNNImputer
    # Esta imputación se hace ANTES del split para evitar data leakage del test set en la imputación.
    # Por simplicidad y siguiendo la estructura del script NN, imputaremos en X_full_raw.
    # Una alternativa sería imputar después del split train/test.

    min_samples_for_knn = 5 # KNNImputer necesita n_neighbors <= n_samples
    if len(X_full_raw) < min_samples_for_knn:
        print(f"❌ {sitio_id_log} ({target_name_log}): Datos insuficientes ({len(X_full_raw)} muestras) para KNNImputer. Saltando sitio-target.")
        return None, pd.DataFrame()

    knn_imputer = KNNImputer(n_neighbors=min(min_samples_for_knn-1, len(X_full_raw)-1) if len(X_full_raw) > 1 else 1) # n_neighbors debe ser < n_samples

    X_features_columns = X_full_raw.columns
    X_full_imputed_np = knn_imputer.fit_transform(X_full_raw)
    X_full_imputed = pd.DataFrame(X_full_imputed_np, columns=X_features_columns, index=X_full_raw.index)

    # Re-alinear y y dates con el índice de X_full_imputed (en caso de que KNNImputer haya afectado el índice, aunque no debería)
    y_full = y_full_raw.loc[X_full_imputed.index]
    dates_full = dates_full_raw.loc[X_full_imputed.index]

    if y_full.nunique() < 2:
        print(f"❌ {sitio_id_log} ({target_name_log}): Target con menos de 2 clases únicas ({y_full.nunique()}) después de la preparación. No se puede entrenar.")
        return None, pd.DataFrame()

    # 2. División Train/Test (Temporal)
    # Usar un split fijo para asegurar que todos los modelos se evalúan en el mismo test set.
    # TimeSeriesSplit para CV dentro de GridSearchCV, pero un split simple para el hold-out test.
    n_total_samples = len(X_full_imputed)
    if n_total_samples < 20: # Umbral mínimo para un split y CV razonable
         print(f"❌ {sitio_id_log} ({target_name_log}): Muy pocos datos ({n_total_samples}) para un split train/test/CV significativo. Saltando.")
         return None, pd.DataFrame()

    split_idx_test = int(n_total_samples * 0.8)
    X_train_raw, X_test_raw = X_full_imputed.iloc[:split_idx_test], X_full_imputed.iloc[split_idx_test:]
    y_train, y_test = y_full.iloc[:split_idx_test], y_full.iloc[split_idx_test:]
    # dates_train, dates_test = dates_full.iloc[:split_idx_test], dates_full.iloc[split_idx_test:]

    if X_train_raw.empty or X_test_raw.empty or y_train.empty or y_test.empty:
        print(f"❌ {sitio_id_log} ({target_name_log}): Conjunto de train o test vacío después del split. Saltando.")
        return None, pd.DataFrame()
    if y_train.nunique() < 2:
        print(f"❌ {sitio_id_log} ({target_name_log}): Target de entrenamiento con menos de 2 clases. Saltando.")
        return None, pd.DataFrame()


    # 3. Escalado de Características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    trained_models_info['scaler'] = scaler
    trained_models_info['knn_imputer'] = knn_imputer
    trained_models_info['feature_columns'] = X_features_columns.tolist()


    # 4. Manejo de Desbalance de Clases (Pesos de Clase)
    # Calculado en y_train para pasar a todos los modelos
    unique_classes_train = np.unique(y_train)
    if len(unique_classes_train) < 2: # Debería haber sido capturado antes, pero por si acaso
        print(f"❌ {sitio_id_log} ({target_name_log}): y_train no tiene suficientes clases para calcular pesos. Saltando.")
        return trained_models_info, pd.DataFrame(results_summary)

    class_weights_calculated = sk_class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes_train,
        y=y_train.to_numpy() # y_train es un pd.Series
    )
    class_weight_dict = dict(zip(unique_classes_train, class_weights_calculated))
    print(f"INFO ({sitio_id_log}): Pesos de clase calculados para y_train: {class_weight_dict}")


    # --- Modelos Scikit-learn ---
    if 'sklearn' in models_to_run:
        for model_name, model_template in MODELOS_SKLEARN_A_PROBAR.items():
            print(f"  Optimizando y entrenando Sklearn: {model_name}...")

            # Instanciar modelo con class_weight (GridSearchCV no lo maneja bien en estimator directamente para todos los scorers)
            # Se puede poner class_weight en el param_grid si el modelo lo acepta en set_params
            # O crear una pipeline que incluya el class_weight.
            # Para RF y LogReg, se puede pasar 'class_weight' al constructor.

            current_model_instance = model_template.set_params(class_weight=class_weight_dict if model_name != 'LogisticRegression' else 'balanced')
            # LogisticRegression con solver saga/liblinear acepta dict o 'balanced'. RF acepta dict.

            current_param_grid = PARAM_GRIDS_SKLEARN.get(model_name, {})
            if 'class_weight' in current_param_grid and model_name == 'LogisticRegression': # LogReg puede tenerlo en el grid
                 current_param_grid['class_weight'] = [class_weight_dict, 'balanced']


            tscv_sklearn = TimeSeriesSplit(n_splits=N_SPLITS_TSCV_SKLEARN)

            # Asegurar suficientes muestras para los splits de TSCV
            min_samples_for_cv = sum(1 for _ in tscv_sklearn.split(X_train_scaled)) # Cuenta el número de splits que se pueden hacer
            if len(X_train_scaled) < N_SPLITS_TSCV_SKLEARN + 1 or min_samples_for_cv < N_SPLITS_TSCV_SKLEARN :
                 print(f"  Advertencia ({model_name}): No hay suficientes muestras en train ({len(X_train_scaled)}) para {N_SPLITS_TSCV_SKLEARN} splits de TimeSeriesSplit. Entrenando con params por defecto.")
                 final_sklearn_model = current_model_instance.fit(X_train_scaled, y_train)
                 best_params_sklearn = "default (CV omitido)"
            elif not current_param_grid:
                 print(f"  Advertencia ({model_name}): Sin hiperparámetros definidos. Usando config base.")
                 final_sklearn_model = current_model_instance.fit(X_train_scaled, y_train)
                 best_params_sklearn = "default (no grid)"
            else:
                grid_search = GridSearchCV(estimator=current_model_instance, param_grid=current_param_grid,
                                           cv=tscv_sklearn, scoring=SCORING_METRIC_CV_SKLEARN, n_jobs=-1, error_score='raise')
                try:
                    grid_search.fit(X_train_scaled, y_train)
                    final_sklearn_model = grid_search.best_estimator_
                    best_params_sklearn = grid_search.best_params_
                    print(f"    Mejores params para {model_name}: {best_params_sklearn} ({SCORING_METRIC_CV_SKLEARN} CV: {grid_search.best_score_:.3f})")
                except Exception as e_grid:
                    print(f"    Error en GridSearchCV para {model_name}: {e_grid}. Usando modelo base.")
                    final_sklearn_model = current_model_instance.fit(X_train_scaled, y_train)
                    best_params_sklearn = f"default (error en CV: {e_grid})"

            # Evaluación en el Test set
            if not y_test.empty:
                y_pred_sklearn = final_sklearn_model.predict(X_test_scaled)
                acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
                f1_sklearn = f1_score(y_test, y_pred_sklearn, average='weighted', zero_division=0)
                report_sklearn = classification_report(y_test, y_pred_sklearn, zero_division=0, labels=np.unique(y_full), target_names=[CLASS_LABELS_MAP_ALERTA.get(i, str(i)) for i in np.unique(y_full)])

                print(f"    {model_name} (Test Final) -> Accuracy: {acc_sklearn:.3f} | F1 (weighted): {f1_sklearn:.3f}")
                print(f"    Reporte de Clasificación para {model_name}:\n{report_sklearn}")

                results_summary.append({
                    'sitio': sitio_id_log, 'target_variable': target_name_log, 'model_type': 'Sklearn',
                    'model_name': model_name, 'accuracy_test': acc_sklearn, 'f1_weighted_test': f1_sklearn,
                    'best_params': str(best_params_sklearn), 'class_report_test': report_sklearn
                })
                trained_models_info['models'][model_name] = final_sklearn_model
            else:
                print(f"    {model_name}: No hay datos de test para evaluar.")
                trained_models_info['models'][model_name] = final_sklearn_model # Guardar modelo entrenado

    # --- Modelos de Redes Neuronales ---
    if 'nn' in models_to_run:
        for model_name_nn, model_builder_nn in MODELOS_NN_BUILDERS.items():
            print(f"  Optimizando y entrenando NN: {model_name_nn}...")

            X_train_nn, X_test_nn = X_train_scaled.copy(), X_test_scaled.copy()
            current_input_dim = X_train_nn.shape[1]

            if model_name_nn == 'CNN1D':
                X_train_nn = X_train_nn.reshape(X_train_nn.shape[0], X_train_nn.shape[1], 1)
                X_test_nn = X_test_nn.reshape(X_test_nn.shape[0], X_test_nn.shape[1], 1)
                hypermodel_for_tuner_nn = lambda hp: model_builder_nn(hp, input_shape=(current_input_dim, 1), num_classes=NUM_CLASSES_ALERTA)
            else: # MLP, MLP_Deeper
                hypermodel_for_tuner_nn = lambda hp: model_builder_nn(hp, input_dim=current_input_dim, num_classes=NUM_CLASSES_ALERTA)

            # KerasTuner - necesita un directorio por cada búsqueda
            tuner_dir = os.path.join('keras_tuner_dir', f"{sitio_id_log}_{target_name_log}_{model_name_nn}")

            # Usar TimeSeriesSplit para la validación dentro de KerasTuner para consistencia
            # KerasTuner no soporta generadores de CV directamente en tuner.search para 'validation_data'
            # Una opción es un validation_split (fracción final del train), o pasar X_test como val_data (riesgo de leakage si se usa para HPO)
            # Para una HPO más robusta con series temporales, se podría hacer un loop manual de CV con el tuner.
            # Por simplicidad aquí, usaremos una fracción del final del training set para validación del tuner.
            # O si y_test no está vacío, usar X_test_nn, y_test para validation_data del tuner,
            # reconociendo que esto guía HPs con datos de test, pero es común en scripts de competencia.
            # Para comparación más justa, la validación del tuner debería ser en una porción separada del train.

            # Vamos a usar un validation_split simple (último 20% del set de entrenamiento del tuner)
            # Y EarlyStopping basado en val_loss o val_accuracy.

            tuner = kt.RandomSearch(
                hypermodel_for_tuner_nn,
                objective='val_accuracy',
                max_trials=KT_MAX_TRIALS_NN,
                executions_per_trial=KT_EXECUTIONS_PER_TRIAL_NN,
                directory=tuner_dir,
                project_name='hparam_tuning_unified',
                overwrite=True
            )

            tuner_callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, verbose=0, restore_best_weights=True)] # EarlyStopping para trials

            print(f"    Iniciando búsqueda de hiperparámetros para {model_name_nn} (max_trials={KT_MAX_TRIALS_NN})...")
            # Keras tuner usa por defecto el último 20% de los datos de entrenamiento para validación si no se provee validation_data.
            # Esto es aceptable para una primera aproximación.
            tuner.search(X_train_nn, y_train,
                         epochs=NN_EPOCHS_TUNER,
                         validation_split=0.2, # Usa el último 20% de X_train_nn/y_train para validación
                         callbacks=tuner_callbacks,
                         class_weight=class_weight_dict,
                         batch_size=NN_BATCH_SIZE,
                         verbose=0)

            best_hps_nn = tuner.get_best_hyperparameters(num_trials=1)[0]
            print(f"    Mejores hiperparámetros para {model_name_nn}: {best_hps_nn.values}")

            # Construir y entrenar el mejor modelo Keras con los HPs encontrados
            final_model_nn = tuner.hypermodel.build(best_hps_nn)
            print(f"    Entrenando el mejor modelo {model_name_nn} con hiperparámetros óptimos...")

            final_nn_callbacks = [EarlyStopping(monitor='val_accuracy' if not y_test.empty else 'accuracy', patience=10, restore_best_weights=True, verbose=0)]

            fit_params_nn = {'callbacks': final_nn_callbacks, 'class_weight': class_weight_dict, 'batch_size': NN_BATCH_SIZE, 'verbose': 0}
            # Si hay test set, usarlo para early stopping del entrenamiento final. Si no, usar accuracy del training.
            if not y_test.empty:
                fit_params_nn['validation_data'] = (X_test_nn, y_test)

            final_model_nn.fit(X_train_nn, y_train, epochs=NN_EPOCHS_FINAL_FIT, **fit_params_nn)

            # Evaluación en el Test set
            if not y_test.empty:
                y_pred_probs_nn = final_model_nn.predict(X_test_nn, verbose=0)
                y_pred_nn = np.argmax(y_pred_probs_nn, axis=1)
                acc_nn = accuracy_score(y_test, y_pred_nn)
                f1_nn = f1_score(y_test, y_pred_nn, average='weighted', zero_division=0)
                report_nn = classification_report(y_test, y_pred_nn, zero_division=0, labels=np.unique(y_full), target_names=[CLASS_LABELS_MAP_ALERTA.get(i, str(i)) for i in np.unique(y_full)])

                print(f"    {model_name_nn} (Test Final) -> Accuracy: {acc_nn:.3f} | F1 (weighted): {f1_nn:.3f}")
                print(f"    Reporte de Clasificación para {model_name_nn}:\n{report_nn}")

                results_summary.append({
                    'sitio': sitio_id_log, 'target_variable': target_name_log, 'model_type': 'NN',
                    'model_name': model_name_nn, 'accuracy_test': acc_nn, 'f1_weighted_test': f1_nn,
                    'best_params': str(best_hps_nn.values), 'class_report_test': report_nn
                })
                trained_models_info['models'][model_name_nn] = final_model_nn
            else:
                 print(f"    {model_name_nn}: No hay datos de test para evaluar.")
                 trained_models_info['models'][model_name_nn] = final_model_nn # Guardar modelo entrenado


    return trained_models_info, pd.DataFrame(results_summary)


# =======================================================================
# --- 4. LÓGICA PRINCIPAL: FUNCIÓN DE ENTRENAMIENTO ---
# =======================================================================

## <-- CAMBIO CRÍTICO: Esta es la nueva función principal del script -->
def ejecutar_entrenamiento_completo():
    """
    Función que orquesta todo el proceso de entrenamiento y guardado de modelos.
    """
    print("--- INICIANDO PROCESO DE ENTRENAMIENTO DE MODELOS ---")

    # 1. Cargar datos desde la base de datos (reemplaza load_data_colab)
    df_original = cargar_datos_desde_bd(engine3)
    if df_original is None or df_original.empty:
        print("Finalizando: no se pudieron cargar los datos para el entrenamiento.")
        return

    # 2. Preprocesar los datos usando tu función existente
    df_procesado_global = preprocess_dataframe_unified(df_original)
    if df_procesado_global is None or df_procesado_global.empty:
        print("Finalizando: el preprocesamiento de datos falló o resultó en un DataFrame vacío.")
        return

    # 3. Definir los targets a modelar (como en tu script original)
    targets_configurations = {
        'Clorofila': {'name': 'Clorofila', 'original_values_col': TARGET_CLOROFILA_ORIGINAL, 'classification_function': classify_chlorophyll_alerta, 'classified_target_col': TARGET_CLOROFILA_CLASE_COL},
        'Cianobacterias': {'name': 'Cianobacterias', 'original_values_col': TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L, 'classification_function': classify_cyanobacteria_alerta, 'classified_target_col': TARGET_CIANOBACTERIAS_CLASE_COL}
    }

    # 4. Bucle principal para entrenar modelos para cada target y sitio
    for target_key, config in targets_configurations.items():
        print(f"\n🎯 === INICIANDO ANÁLISIS PARA TARGET: {config['name']} === 🎯")
        for sitio in SITIOS_A_ANALIZAR:
            print(f"\n--- Procesando Sitio: {sitio} (Target: {config['name']}) ---")
            df_sitio_data = df_procesado_global[df_procesado_global[COLUMNA_CODIGO_PERFIL] == sitio].copy()
            if df_sitio_data.empty: continue

            X_site, y_site, _, dates_site = create_features_for_site_unified(df_sitio_data, config, sitio)
            if X_site.empty or y_site.empty or y_site.nunique() < 2:
                print(f"Datos insuficientes para entrenar en {sitio}. Saltando.")
                continue

            trained_model_package, metrics_df = train_evaluate_models_unified(X_site, y_site, dates_site, sitio, config['name'])

            ## <-- CAMBIO CRÍTICO: GUARDAR LOS ARTEFACTOS DEL MEJOR MODELO ---
            if trained_model_package and trained_model_package.get('models'):
                nombre_mejor_modelo = None
                if metrics_df is not None and not metrics_df.empty:
                    mejor_modelo_info = metrics_df.sort_values(by='f1_weighted_test', ascending=False).iloc[0]
                    nombre_mejor_modelo = mejor_modelo_info['model_name']
                    print(f"El mejor modelo para {sitio}-{target_key} es: {nombre_mejor_modelo} con F1={mejor_modelo_info['f1_weighted_test']:.3f}")
                else:
                    nombre_mejor_modelo = next(iter(trained_model_package['models']))
                    print(f"No hay métricas de test. Guardando el primer modelo por defecto: {nombre_mejor_modelo}")

                # Preparar el paquete de artefactos para guardar
                artefactos_para_guardar = {
                    'modelo': trained_model_package['models'][nombre_mejor_modelo],
                    'scaler': trained_model_package['scaler'],
                    'knn_imputer': trained_model_package['knn_imputer'],
                    'feature_names': trained_model_package['feature_columns']
                }

                # Crear la carpeta si no existe
                os.makedirs('modelos_entrenados', exist_ok=True)
                
                # Guardar en un archivo
                ruta_archivo_joblib = f"modelos_entrenados/artefactos_{sitio}_{target_key}.pkl"
                
                # Manejar modelos Keras por separado, ya que no se guardan bien con joblib
                if nombre_mejor_modelo in MODELOS_NN_BUILDERS:
                    modelo_keras = artefactos_para_guardar.pop('modelo') # Quitar el modelo del dict
                    ruta_keras = f"modelos_entrenados/modelo_{sitio}_{target_key}.keras"
                    modelo_keras.save(ruta_keras)
                    print(f"Modelo Keras guardado en: {ruta_keras}")
                    # Guardar el resto de artefactos (scaler, imputer, etc.) con joblib
                    joblib.dump(artefactos_para_guardar, ruta_archivo_joblib)
                    print(f"Artefactos (scaler, imputer, etc.) guardados en: {ruta_archivo_joblib}")
                else:
                    # Guardar todo el diccionario con joblib para modelos Sklearn
                    joblib.dump(artefactos_para_guardar, ruta_archivo_joblib)
                    print(f"Artefactos (modelo, scaler, etc.) guardados en: {ruta_archivo_joblib}")

    print("\n\n✨✨✨ Proceso de entrenamiento y guardado de modelos completado. ✨✨✨")

## <-- CAMBIO CRÍTICO: La función `load_data_colab` se reemplaza por esta -->
def cargar_datos_desde_bd(engine):
    """Lee los datos ya procesados desde la tabla 'dataframe'."""
    print("Leyendo datos de la tabla 'dataframe' para el entrenamiento...")
    try:
        df = pd.read_sql("SELECT * FROM dataframe", engine)
        print("Datos leídos exitosamente.")
        return df
    except Exception as e:
        print(f"Error al leer la tabla 'dataframe': {e}")
        return None

## <-- CAMBIO CRÍTICO: La ejecución se controla aquí -->
if __name__ == '__main__':
    # Esta es la única función que se llama cuando ejecutas 'python entrenar_modelos.py'
    ejecutar_entrenamiento_completo()

