import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import warnings
import os
import io
from sqlalchemy import create_engine
import joblib

# Configuración de conexión a BD para entrenamiento
usuario = 'postgres'
contraseña = 'postgres'
host2 = '192.168.191.164'
puerto2 = '5433'
nombre_base_modelo = 'model_data'
engine3 = create_engine(f'postgresql+psycopg2://{usuario}:{contraseña}@{host2}:{puerto2}/{nombre_base_modelo}')

# --- 1. Configuraciones Globales ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
tf.get_logger().setLevel('ERROR')

SITIOS_A_ANALIZAR = ['C1', 'TAC1', 'TAC4', 'DSA1', 'DCQ1']
N_FEATURES_A_SELECCIONAR = 25
N_SPLITS_CV_GRIDSEARCH = 2
N_SPLITS_CV_EVAL = 3
SCORING_METRIC = 'f1_macro'
KT_MAX_TRIALS_NN = 5
NN_EPOCHS_TUNER = 25
NN_EPOCHS_FINAL_FIT = 50
NN_BATCH_SIZE = 16
COLUMNA_FECHA = 'fecha'
COLUMNA_CODIGO_PERFIL = 'codigo_perfil'
TARGET_CLOROFILA_ORIGINAL = 'Clorofila (µg/l)'
TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L = 'Cianobacterias Total'
COLUMNA_DOMINANCIA_CIANO = 'Dominancia de Cianobacterias (%)'
TARGET_DOMINANCIA_CLASE_COL = 'Dominancia_Clase_Alerta'
TARGET_CLOROFILA_CLASE_COL = 'Clorofila_Clase_Alerta'
TARGET_CIANOBACTERIAS_CLASE_COL = 'Cianobacterias_Clase_Alerta'
FEATURES_PARA_LAGS_Y_TENDENCIAS = sorted([
    'Clorofila (µg/l)',
    'Cianobacterias_cel_mL_Calculado',
    'Total_Algas_cel_mL_Calculado',
    'Dominancia de Cianobacterias (%)',
    'T° (°C)',
    'PHT (µg/l)',
    'PRS (µg/l)',
    'Nitrogeno Inorganico Total (µg/l)',
    'Cota (m)',
    'temperatura_max',
    'temperatura_min',
    '600',
    '700',
    '1100'
])

# --- 2. Modelos y Parrillas de Búsqueda ---
lr_base = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
MODELOS_SKLEARN_A_PROBAR = {'LogisticRegression': lr_base, 'RandomForest': rf_base}
PARAM_GRIDS_SKLEARN = {
    'LogisticRegression': {'C': [0.1, 1.0, 10.0]},
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [5, 10]}
}

# --- 3. Funciones Auxiliares ---
def classify_chlorophyll_alerta(v):
    if pd.isna(v): return np.nan
    return 0 if v < 10 else 1 if v <= 24 else 2

def classify_cyanobacteria_alerta(v):
    if pd.isna(v): return np.nan
    v_ml = v / 1000.0
    return 0 if v_ml < 5000 else 1 if v_ml <= 60000 else 2

def classify_dominance_alerta(v):
    if pd.isna(v): return np.nan
    return 0 if v < 50 else 1

def load_data(engine3):
    print("Conectando a la base de datos para leer la tabla 'dataframe'...")
    try:
        df = pd.read_sql("SELECT * FROM dataframe", engine3)
        print(f"Datos cargados exitosamente desde la base de datos. {df.shape[0]} registros encontrados.")
        return df
    except Exception as e:
        print(f"ERROR: No se pudo leer de la base de datos. Motivo: {e}")
        return None

def preprocess_and_feature_engineer(df_raw):
    if df_raw is None: return None
    print("Iniciando preprocesamiento y creación de características...")
    df = df_raw.rename(
        columns={
            'T¬∞ (¬∞C)': 'T° (°C)',
            'Clorofila (¬µg/l)': TARGET_CLOROFILA_ORIGINAL,
            'Total Algas Sumatoria (Cel/L)': 'Total Algas Sumatoria (Cel/L)'
        }
    ).copy()
    df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA], errors='coerce')
    df.dropna(subset=[COLUMNA_FECHA], inplace=True)
    df = df.sort_values([COLUMNA_CODIGO_PERFIL, COLUMNA_FECHA])
    # dummies condicion termica
    if 'condicion_termica' in df.columns:
        df['condicion_termica'].fillna('DESCONOCIDO', inplace=True)
        dummies = pd.get_dummies(df['condicion_termica'], prefix='cond_termica', dtype=int)
        df = pd.concat([df, dummies], axis=1)
    numeric_cols = [
        TARGET_CLOROFILA_ORIGINAL,
        'Total Algas Sumatoria (Cel/L)',
        COLUMNA_DOMINANCIA_CIANO,
        'T° (°C)',
        'PHT (µg/l)',
        'PRS (µg/l)',
        'Nitrogeno Inorganico Total (µg/l)',
        TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Cianobacterias_cel_mL_Calculado'] = df.get(
        TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L,
        pd.Series(index=df.index)
    ) / 1000.0
    df['Total_Algas_cel_mL_Calculado'] = df.get(
        'Total Algas Sumatoria (Cel/L)',
        pd.Series(index=df.index)
    ) / 1000.0
    df['mes_sin'] = np.sin(2 * np.pi * df[COLUMNA_FECHA].dt.month / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df[COLUMNA_FECHA].dt.month / 12)
    new_cols = []
    for col in FEATURES_PARA_LAGS_Y_TENDENCIAS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            grouped = df.groupby(COLUMNA_CODIGO_PERFIL)[col]
            for lag in [1, 2, 3]:
                new_cols.append(
                    grouped.shift(lag).rename(f'{col}lag{lag}')
                )
            for window in [3, 6]:
                new_cols.append(
                    grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
                           .rename(f'{col}roll_mean{window}')
                )
                new_cols.append(
                    grouped.transform(lambda x: x.rolling(window, min_periods=1).std())
                           .rename(f'{col}roll_std{window}')
                )
    df = pd.concat([df] + new_cols, axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def prepare_final_dataset(df_site_input, target_config):
    df_site = df_site_input.sort_values(by=COLUMNA_FECHA).copy()
    record_for_prediction = df_site.iloc[-1:].copy()
    col_orig = target_config['original_values_col']
    df_site[target_config['classified_target_col']] = (
        df_site[col_orig].apply(target_config['classification_function'])
    )
    y = df_site[target_config['classified_target_col']].shift(-1)
    y.name = 'target'
    drops = [
        'id_registro', COLUMNA_FECHA, COLUMNA_CODIGO_PERFIL,
        TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L,
        'Total Algas Sumatoria (Cel/L)',
        TARGET_CLOROFILA_CLASE_COL, TARGET_CIANOBACTERIAS_CLASE_COL,
        TARGET_DOMINANCIA_CLASE_COL,
        'condicion_termica'
    ]
    X = df_site.drop(columns=[c for c in drops if c in df_site.columns], errors='ignore')
    full = pd.concat([X, y], axis=1).dropna(subset=['target'])
    if full.empty:
        return pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame()
    X_final = full.drop(columns='target')
    y_final = full['target'].astype(int)
    X_for_prediction = record_for_prediction.reindex(
        columns=X_final.columns, fill_value=0
    )
    return X_final, y_final, X_for_prediction

def select_best_features(X, y, n_features):
    Xn = X.select_dtypes(include=np.number).replace([np.inf, -np.inf], 0).fillna(0)
    if Xn.shape[1] <= n_features:
        return Xn.columns.tolist()
    sel = SelectKBest(mutual_info_classif, k=n_features)
    sel.fit(Xn, y)
    return sel.get_feature_names_out().tolist()

def build_mlp_model_for_tuner(hp, input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units=hp.Int('units', 32, 256, 32), activation='relu'),
        BatchNormalization(),
        Dropout(rate=hp.Float('dropout', 0.2, 0.5, 0.1)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-3, 5e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_final_mlp_model(params, input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units=params['units'], activation='relu'),
        BatchNormalization(),
        Dropout(rate=params['dropout']),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='sparse_categorical_crossentropy', metrics=['accuracy']
    )
    return model

def train_evaluate_sklearn(X, y, sitio, target_name):
    print(f"--- Procesando Modelos Sklearn: Sitio {sitio}, Objetivo {target_name} ---")
    if len(X) < 20 or y.nunique() < 2:
        return pd.DataFrame()
    results = []
    for name, tmpl in MODELOS_SKLEARN_A_PROBAR.items():
        try:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            grid = GridSearchCV(
                tmpl, PARAM_GRIDS_SKLEARN.get(name, {}),
                cv=TimeSeriesSplit(n_splits=N_SPLITS_CV_GRIDSEARCH),
                scoring=SCORING_METRIC, n_jobs=-1, error_score=0.0
            )
            grid.fit(Xs, y)
            best = grid.best_params_
            print(f"  -> {name}: Mejores Parámetros: {best}")
            f1s, rocs = [], []
            tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV_EVAL)
            model = tmpl.set_params(**best)
            for tr, te in tscv.split(X):
                X_tr, X_te = X.iloc[tr], X.iloc[te]
                y_tr, y_te = y.iloc[tr], y.iloc[te]
                if y_tr.nunique() < 2: continue
                sf = StandardScaler().fit(X_tr)
                Xtr_s, Xte_s = sf.transform(X_tr), sf.transform(X_te)
                model.fit(Xtr_s, y_tr)
                y_pred = model.predict(Xte_s)
                f1s.append(f1_score(y_te, y_pred, average='macro', zero_division=0))
                roc = np.nan
                if hasattr(model, 'predict_proba') and y_te.nunique()>1:
                    probs = model.predict_proba(Xte_s)
                    labels = np.unique(np.concatenate([y_tr, y_te]))
                    if len(labels)>2:
                        roc = roc_auc_score(y_te, probs, multi_class='ovr', average='weighted', labels=labels)
                    else:
                        roc = roc_auc_score(y_te, probs[:,1])
                rocs.append(roc)
            if f1s:
                results.append({
                    'sitio': sitio,
                    'target': target_name,
                    'modelo': name,
                    'f1_macro_cv': np.nanmean(f1s),
                    'roc_auc_cv': np.nanmean(rocs),
                    'best_params': best
                })
        except Exception as e:
            print(f"  -> ❌ ERROR en {name} para {sitio}/{target_name}: {e}")
    return pd.DataFrame(results)

def train_evaluate_mlp(X, y, sitio, target_name, tuner_dir):
    print(f"--- Procesando MLP: Sitio {sitio}, Objetivo {target_name} ---")
    if len(X) < 30 or y.nunique()<2:
        return pd.DataFrame()
    try:
        tscv = TimeSeriesSplit(n_splits=2)
        train_val_idx, test_idx = list(tscv.split(X))[-1]
        X_tv, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
        y_tv, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
        if X_tv.empty or y_tv.nunique()<2:
            return pd.DataFrame()
        tscv2 = TimeSeriesSplit(n_splits=2)
        tr, val = list(tscv2.split(X_tv))[-1]
        X_tr, X_val = X_tv.iloc[tr], X_tv.iloc[val]
        y_tr, y_val = y_tv.iloc[tr], y_tv.iloc[val]
        if X_tr.empty or y_tr.nunique()<2:
            return pd.DataFrame()
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_val_s, X_test_s = scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_test)
        num_classes = len(np.unique(y))
        
        tuner = kt.RandomSearch(
            lambda hp: build_mlp_model_for_tuner(hp, X.shape[1], num_classes),
            objective='val_accuracy', max_trials=KT_MAX_TRIALS_NN,
            
            # Usamos la ruta principal para el directorio
            directory=tuner_dir,
            
            # Usamos un nombre de proyecto único para la subcarpeta
            project_name=f'kt_{sitio}_{target_name}',
            
            overwrite=True
        )
        tuner.search(
            X_tr_s, y_tr, epochs=NN_EPOCHS_TUNER,
            validation_data=(X_val_s, y_val),
            callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
            verbose=0
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        final = tuner.hypermodel.build(best_hps)
        final.fit(
            np.vstack([X_tr_s, X_val_s]), np.concatenate([y_tr, y_val]),
            epochs=NN_EPOCHS_FINAL_FIT,
            batch_size=NN_BATCH_SIZE,
            verbose=0
        )
        y_proba = final.predict(X_test_s, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        roc = np.nan
        if y_test.nunique()>1:
            try:
                labs = np.unique(np.concatenate([y_tv, y_test]))
                if num_classes>2:
                    roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted', labels=labs)
                else:
                    roc = roc_auc_score(y_test, y_proba[:,1])
            except ValueError:
                pass
        return pd.DataFrame([{  
            'sitio': sitio,
            'target': target_name,
            'modelo': 'MLP',
            'f1_macro_cv': f1,
            'roc_auc_cv': roc,
            'best_params': best_hps.values
        }])
    except Exception as e:
        print(f"  -> ❌ ERROR en MLP para {sitio}/{target_name}: {e}")
        return pd.DataFrame()

# --- 4. Función Principal de Ejecución ---
def main():
    """
    Función principal que orquesta todo el proceso:
    1. Carga de datos desde la Base de Datos.
    2. Preprocesa y crea características para todos los datos.
    3. Define los directorios de salida para modelos y logs del tuner.
    4. Itera por cada objetivo y sitio de monitoreo.
    5. Para cada combinación, evalúa múltiples modelos y selecciona el mejor.
    6. Re-entrena el mejor modelo con todos los datos disponibles.
    7. Guarda los artefactos finales (modelo, scaler, lista de features) en el disco.
    8. Imprime un resumen final de todos los resultados.
    """
    # 1. Carga de datos
    df_raw = load_data(engine3)
    if df_raw is None:
        print("Finalizando: no se pudieron cargar los datos.")
        return

    # 2. Preprocesamiento general
    df_processed = preprocess_and_feature_engineer(df_raw)

    # 3. Definición de directorios de salida
    DIR_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos_entrenados')
    os.makedirs(DIR_MODELOS, exist_ok=True)
    
    DIR_TUNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuner_logs')
    # No es necesario crear DIR_TUNER, Keras Tuner lo maneja.

    # Configuración de los targets a modelar
    targets_config = {
        'Clorofila': {
            'original_values_col': TARGET_CLOROFILA_ORIGINAL,
            'classification_function': classify_chlorophyll_alerta,
            'classified_target_col': TARGET_CLOROFILA_CLASE_COL
        },
        'Cianobacterias': {
            'original_values_col': TARGET_CIANOBACTERIAS_ORIGINAL_CEL_L,
            'classification_function': classify_cyanobacteria_alerta,
            'classified_target_col': TARGET_CIANOBACTERIAS_CLASE_COL
        },
        'Dominancia': {
            'original_values_col': COLUMNA_DOMINANCIA_CIANO,
            'classification_function': classify_dominance_alerta,
            'classified_target_col': TARGET_DOMINANCIA_CLASE_COL
        }
    }
    
    all_results = []
    # 4. Bucle principal por Target y Sitio
    for target_key, cfg in targets_config.items():
        print(f"\n🎯 === PROCESANDO OBJETIVO: {target_key} === 🎯")
        for sitio in SITIOS_A_ANALIZAR:
            print(f"\n--- Procesando {sitio} / {target_key} ---")
            
            df_site = df_processed[df_processed[COLUMNA_CODIGO_PERFIL] == sitio]
            if df_site.empty:
                print(f"INFO: Omitiendo. No hay datos para el sitio {sitio}.")
                continue
            
            X, y, _ = prepare_final_dataset(df_site, cfg) # No necesitamos X_pred aquí
            if X.empty or y.nunique() < 2:
                print(f"INFO: Omitiendo. Datos insuficientes para entrenar (N de clases < 2 o data vacía).")
                continue
            
            best_feats = select_best_features(X, y, N_FEATURES_A_SELECCIONAR)
            if not best_feats:
                print(f"INFO: Omitiendo. No se seleccionaron características.")
                continue

            print(f"  -> Features seleccionadas: {len(best_feats)}")
            X_sel = X[best_feats]
            
            # 5. Evaluación de modelos
            res_skl = train_evaluate_sklearn(X_sel, y, sitio, target_key)
            res_mlp = train_evaluate_mlp(X_sel, y, sitio, target_key, DIR_TUNER) # Pasamos dir del tuner
            cur = pd.concat([res_skl, res_mlp], ignore_index=True)
            
            if cur.empty:
                print(f"  -> ❌ No se pudo evaluar ningún modelo para esta combinación.")
                continue
            
            all_results.append(cur)
            
            # 6. Re-entrenamiento del mejor modelo
            best_row = cur.loc[cur['f1_macro_cv'].idxmax()]
            model_name, params = best_row['modelo'], best_row['best_params']
            
            print(f"💾 Entrenando y guardando el mejor modelo: {model_name} (F1-Score CV: {best_row['f1_macro_cv']:.4f})")
            print(f"  -> Mejor modelo: {model_name} (F1-Score CV: {best_row['f1_macro_cv']:.4f})")

            final_model = None
            scaler = StandardScaler().fit(X_sel)
            X_train_full_scaled = scaler.transform(X_sel)

            if model_name in MODELOS_SKLEARN_A_PROBAR:
                mdl_template = MODELOS_SKLEARN_A_PROBAR[model_name]
                final_model = mdl_template.set_params(**params)
                final_model.fit(X_train_full_scaled, y)
            
            elif model_name == 'MLP':
                num_classes = y.nunique()
                final_model = create_final_mlp_model(params, X_sel.shape[1], num_classes)
                final_model.fit(X_train_full_scaled, y, epochs=NN_EPOCHS_FINAL_FIT, batch_size=NN_BATCH_SIZE, verbose=0)

            # 7. Guardado de artefactos finales
            if final_model:
                artefactos_para_guardar = {
                    'scaler': scaler,
                    'best_features': best_feats,
                    'model_info': best_row.to_dict()
                }
                
                ruta_joblib = os.path.join(DIR_MODELOS, f"artefactos_{sitio}_{target_key}.pkl")

                if model_name == 'MLP':
                    ruta_keras = os.path.join(DIR_MODELOS, f"modelo_{sitio}_{target_key}.keras")
                    final_model.save(ruta_keras)
                    joblib.dump(artefactos_para_guardar, ruta_joblib)
                else: 
                    artefactos_para_guardar['modelo'] = final_model
                    joblib.dump(artefactos_para_guardar, ruta_joblib)
                
                print(f"  -> ✅ Artefactos guardados exitosamente.")
            else:
                print(f"  -> ❌ ERROR: No se pudo entrenar el modelo final para guardar.")

    # 8. Impresión del resumen final
    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        print("\n\n📊📊📊 RESUMEN FINAL DE MÉTRICAS DE VALIDACIÓN CRUZADA 📊📊📊")
        cols_to_show = ['target', 'sitio', 'modelo', 'f1_macro_cv', 'roc_auc_cv', 'best_params']
        summary_display = summary.reindex(columns=cols_to_show, fill_value='-')
        print(summary_display.sort_values(by=['target', 'sitio', 'f1_macro_cv'], ascending=[True, True, False]).to_string(index=False))
        
    print("\n✨ Proceso de entrenamiento y guardado completado. ✨")

if __name__ == '__main__':
    main()

