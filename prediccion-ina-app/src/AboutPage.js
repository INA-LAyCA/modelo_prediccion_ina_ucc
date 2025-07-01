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
            <h2>¿Qué es este Modelo?</h2>
          </div>
          <p>
            Este sistema es una herramienta de <strong>alerta temprana</strong> desarrollada para el Instituto Nacional del Agua (INA-CIRSA) para el monitoreo y predicción de la calidad del agua en puntos estratégicos de los embalses.
          </p>
          <p>
            Su objetivo principal es <strong>anticipar con un mes de antelación</strong> la probabilidad de que ocurran eventos que comprometan la calidad del recurso hídrico, tales como floraciones de algas, altos niveles de clorofila o una dominancia de cianobacterias, un tipo de microalga de especial interés sanitario.
          </p>
          <p>
            Mediante el análisis de series de tiempo históricas y la aplicación de modelos de inteligencia artificial (Machine Learning), esta herramienta identifica patrones complejos para pronosticar el estado futuro del agua, facilitando una gestión más proactiva y eficiente.
          </p>
        </div>

        <div className="about-card">
          <div className="card-header">
            <Icon><PurposeIcon /></Icon>
            <h2>Propósito y Aplicaciones</h2>
          </div>
          <p>
            La capacidad de predecir estos eventos permite a los gestores de recursos hídricos y a las autoridades sanitarias tomar decisiones informadas y oportunas, con aplicaciones como:
          </p>
          <ul>
            <li><strong>Gestión de Plantas Potabilizadoras:</strong> Permite a los operadores anticipar cambios en la calidad del agua cruda para ajustar preventivamente los procesos de tratamiento, asegurando la inocuidad del agua potable y optimizando costos operativos.</li>
            <li><strong>Salud Pública:</strong> Alerta sobre condiciones que podrían representar un riesgo para el uso recreativo de los embalses (natación, deportes acuáticos), ayudando a proteger a la población.</li>
            <li><strong>Monitoreo Ambiental:</strong> Funciona como un sistema de vigilancia inteligente que complementa el monitoreo tradicional, permitiendo enfocar los recursos de muestreo en los momentos y lugares de mayor riesgo pronosticado.</li>
          </ul>
        </div>
        
        <div className="about-card">
          <div className="card-header">
            <Icon><ReadingIcon /></Icon>
            <h2>Interpretación de las Predicciones</h2>
          </div>
          <p>
            El modelo clasifica el riesgo futuro en tres niveles de alerta, siguiendo un esquema de semáforo:
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
                  <strong>Emergencia:</strong> Se pronostica que los niveles superarán umbrales críticos, con alta probabilidad de una floración algal intensa. Requiere acciones inmediatas.
                </div>
              </div>
            </div>
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
        </div>
      </div>
  );
}

export default AboutPage;
