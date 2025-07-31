import React from 'react';
import { useNavigate } from 'react-router-dom';
import './AboutPage.css'; 
  
const Icon = ({ children }) => (
  <div className="icon-container">{children}</div>
);

const ModelIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path></svg>
);
const PurposeIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="4"></circle><line x1="21.17" y1="8" x2="12" y2="8"></line><line x1="3.95" y1="6.06" x2="8.54" y2="14"></line><line x1="10.88" y1="21.94" x2="15.46" y2="14"></line></svg>
);
const ReadingIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>
);
const MetricsIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
);

const LocationIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>
);

const sitiosDeMonitoreo = [

  { termino: 'C', nombre: 'Centro del Embalse', descripcion: 'Punto de muestreo central, representativo de la zona principal del embalse.' },

  { termino: 'TAC', nombre: 'Toma de Aguas Cordobesas', descripcion: 'Ubicado cerca de una de las principales tomas de agua para potabilización.' },

  { termino: 'DCQ', nombre: 'Desembocadura Río Cosquín', descripcion: 'Punto de control en la zona de mezcla donde el Río Cosquín ingresa al embalse.' },

  { termino: 'DSA', nombre: 'Desembocadura Río San Antonio', descripcion: 'Punto de control en la zona de mezcla donde el Río San Antonio, ingresa al embalse.' },

];

function AboutPage() {
  const navigate = useNavigate();

  return (
    <div className="about-container">
      <div className="about-header">
        <button onClick={() => navigate('/')} className="back-button">&larr;</button>
        <h1 className="about-title">Acerca del Modelo de Predicción</h1>
      </div>
      
      <div className="about-content">
        <div className="about-card">
          <div className="card-header">
            <Icon><ModelIcon /></Icon>
            <h2>¿Qué es?</h2>
          </div>
          <p>
            Este sistema es una herramienta de <strong>alerta temprana</strong> desarrollada para el Instituto Nacional del Agua (INA-CIRSA) para el monitoreo y predicción de la calidad del agua en puntos estratégicos del Embalse San Roque.
          </p>
          <p>
            Su objetivo principal es <strong>realizar una predicción, con un mes de antelación</strong> sobre la probabilidad de que ocurran eventos que comprometan la calidad natural del agua del embalse, tales como floraciones de algas, altos niveles de clorofila o una dominancia de cianobacterias, un tipo de microalga de especial interés sanitario.
          </p>
          <p>
            Mediante el análisis de series de tiempo históricas y la aplicación de modelos de inteligencia artificial (Machine Learning), esta herramienta identifica patrones complejos para predecir el estado futuro del agua, facilitando una gestión más proactiva y eficiente.
          </p>
          <p>
            Para lograr estas predicciones, el sistema entrena, evalúa y compara automáticamente un conjunto de modelos de clasificación, incluyendo <strong>Regresión Logística</strong>, <strong>Random Forest</strong> y <strong>Red Neuronal Multicapa (MLP)</strong>.
          </p>
          <p>
            Para más detalles: 
          </p>
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
            {/* El atributo 'download' le dice al navegador que descargue el archivo en lugar de navegar a él */}
            <a 
              href={process.env.PUBLIC_URL + '/informe_tecnico.pdf'} 
              download="Informe de Trabajo Final - Modelo de Predicción.pdf"
              className="primary-button"
            >
              Descargar Informe de Trabajo Final - Modelo de Predicción (PDF)
            </a>
          </div>
        </div>

        <div className="about-card">
          <div className="card-header">
            <Icon><PurposeIcon /></Icon>
            <h2>Propósito y Aplicaciones</h2>
          </div>
          <p>
            La capacidad de predecir estos eventos permite a los organismos responsables o con competencia en la gestión, tomar decisiones informadas y oportunas, en el ambito de:
          </p>
          <ul>
            <li><strong>Gestión de Plantas Potabilizadoras:</strong> Permite a los operadores anticipar cambios en la calidad del agua cruda para ajustar preventivamente los procesos de tratamiento, asegurando la inocuidad del agua potable y optimizando costos operativos.</li>
            <li><strong>Salud Pública:</strong> Alerta sobre condiciones que podrían representar un riesgo para el uso recreativo de los embalses (natación, deportes acuáticos), ayudando a proteger a la población.</li>
            <li><strong>Monitoreo Ambiental:</strong> Funciona como un sistema de vigilancia inteligente que complementa el monitoreo tradicional, permitiendo enfocar los recursos de muestreo en los momentos y lugares de mayor riesgo predicho.</li>
          </ul>
        </div>
        
        <div className="about-card">
            <div className="card-header">
                <Icon><LocationIcon /></Icon>
                <h2>Sitios de Monitoreo</h2>
            </div>
            <p>
                Las predicciones del modelo se basan en datos históricos recopilados en puntos estratégicos del Embalse San Roque.
            </p>
            <img 
                src={process.env.PUBLIC_URL + '/sitios.ong.jpeg'} // Corregido el nombre del archivo si es .png
                alt="Vista del Embalse San Roque"
                style={{ width: '30%', borderRadius: '10px', boxShadow: '0 4px 8px rgba(0,0,0,0.1)', margin: '15px 0' }}
            />
            <div className="table-container">
                <table style={{ margin: '0', width: '100%' }}>
                    <thead>
                        <tr>
                            <th>Término</th>
                            <th>Nombre Completo</th>
                            <th>Descripción</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sitiosDeMonitoreo.map(sitio => (
                            <tr key={sitio.termino}>
                                <td style={{ fontWeight: 'bold' }}>{sitio.termino}</td>
                                <td>{sitio.nombre}</td>
                                <td style={{ textAlign: 'left' }}>{sitio.descripcion}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>

        <div className="about-card">
          <div className="card-header">
            <Icon><ReadingIcon /></Icon>
            <h2>Interpretación de las Predicciones</h2>
          </div>
          <p>
            El modelo clasifica el riesgo futuro en tres niveles de alerta, siguiendo un esquema de semáforo para aguas recreativas propuesto por CARU (Acta Nº09/24)”:
          </p>
          <div className="legend">
              {/* Item Vigilancia */}
              <div className="legend-item">
                <span className="legend-color green"></span>
                <div className="legend-text">
                  <strong>Vigilancia:</strong> Condiciones normales. Se recomienda el monitoreo de rutina.
                </div>
              </div>

              {/* Item Alerta */}
              <div className="legend-item">
                <span className="legend-color yellow"></span>
                <div className="legend-text">
                  <strong>Alerta:</strong> Se prevé que los parámetros superen los umbrales recomendados. Requiere un aumento en la frecuencia de monitoreo y preparación para posibles acciones.
                </div>
              </div>

              {/* Item Emergencia */}
              <div className="legend-item">
                <span className="legend-color red"></span>
                <div className="legend-text">
                  <strong>Emergencia:</strong> Se predice si la dominancia (porcentaje de cianobacterias en relación al total de algas) superara el 50%.
                </div>
              </div>
              {/* Item No dominante */}
              <div className="legend-item">
                <span className="legend-color green"></span>
                <div className="legend-text">
                  <strong>No Dominante:</strong> Se predice si la dominancia (porcentaje de cianobacterias en relación al total de algas) no superara el 50%.
                </div>
              </div>
              {/* Item Dominante */}
              <div className="legend-item">
                <span className="legend-color red"></span>
                <div className="legend-text">
                  <strong>Dominante:</strong>Se predice que el porcentaje de dominancia de cianobacterias sobre el total de algas superara el 50%.
                </div>
              </div>
            </div>
            <div className="threshold-title">Umbrales de Referencia</div>
          <table className="threshold-table">
            <thead>
              <tr>
                <th></th>
                <th className="th-vigilancia">Vigilancia</th>
                <th className="th-alerta1">Alerta</th>
                <th className="th-alerta2">Emergencia</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Clorofila a</strong></td>
                <td>&lt;10 µg/L</td>
                <td>10-24 µg/L</td>
                <td>&gt;24 µg/L</td>
              </tr>
              <tr>
                <td><strong>Cianobacterias Totales</strong></td>
                <td>&lt;5.000 cel/mL</td>
                <td>5.000 - 60.000 cel/mL</td>
                <td>&gt;60.000 cel/mL</td>
              </tr>
            </tbody>
          </table>
          <table className="threshold-table">
            <thead>
              <tr>
                <th></th>
                <th className="th-vigilancia">No Dominante</th>
                <th className="th-alerta2">Dominante</th>
                
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: '1.2' }}>
                    <strong>Dominancia de Cianobacterias</strong>
                    <span style={{ fontSize: '0.8em', color: '#6c757d', marginTop: '4px' }}>
                      Ciano (cel/L) / Algas Totales (cel/L)
                    </span>
                  </div></td>
                <td>&lt;50%</td>
                <td>&gt;=50%</td>
              </tr>
            </tbody>
          </table>
          </div>
        </div>
        <div className="legend">
        </div>
        <div className="about-card">
          <div className="card-header">
            <Icon><MetricsIcon /></Icon>
            <h2>Entendiendo las Métricas del Modelo</h2>
          </div>
          <p>
            Junto a cada predicción se muestran métricas que indican su confiabilidad técnica, validadas mediante un proceso de Cross-Validation (CV).
          </p>
          <h4>F1-Score (CV): Precisión General</h4>
          <p>
            Es una métrica que balancea la precisión (de las veces que predijo "Alerta", cuántas acertó) y la exhaustividad (de todas las "Alertas" reales, cuántas encontró). Un valor más cercano a <strong>1.0</strong> indica un mejor y más equilibrado rendimiento.
          </p>
          <h4>ROC AUC (CV): Capacidad de Discriminación</h4>
          <p>
            Mide la habilidad del modelo para distinguir correctamente entre las diferentes clases (ej. "Alerta" vs. "Vigilancia"). Un valor de 0.5 es aleatorio, mientras que un valor cercano a <strong>1.0</strong> indica que el modelo es muy seguro y consistente en su capacidad para diferenciar los niveles de riesgo.
          </p>
          <h4>Precision Weighted (CV): Precisión Ponderada</h4>
          <p>
            Promedia la precisión de cada clase ponderando por su frecuencia, lo que la hace útil en datos desbalanceados. Valores cercanos a 1.0 indican predicciones más correctas considerando la distribución real de las clases.
          </p>
        </div>
      </div>
  );
}

export default AboutPage;
