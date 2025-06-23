// Dependencias necesarias
// En el directorio del proyecto, instalar las siguientes dependencias:
// npm install axios react-router-dom

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import './App.css';
import logo from './logo.png';

// Componente de pantalla de inicio
function Home({ onPredictStart, onDataStart, onPredictionsStart }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <img src={logo} alt="Logo" className="logo1" />
      <h1>MODELO DE PREDICCIÓN - INA CIRSA</h1>
      <div style={{ display: 'flex', gap: '20px', marginTop: '20px' }}>
        <button onClick={onPredictStart} className="primary-button">Predecir</button>
        <button onClick={onDataStart} className="primary-button">Ver Datos</button>
        <button onClick={onPredictionsStart} className="primary-button">Ver Predicciones</button>
      </div>
    </div>
  );
}

function PredictionDetails({ targetName, data }) {
  // Caso 1: No hay datos para este target en la respuesta de la API.
  if (!data) {
    return (
      <li className="prediction-item">
        <strong>{targetName}:</strong>
        <span className="prediction-label-nodata">No disponible</span>
      </li>
    );
  }

  // Caso 2: 'data' es un string simple (ej: "Modelo no disponible").
  if (typeof data !== 'object') {
    return (
      <li className="prediction-item">
        <strong>{targetName}:</strong>
        <span className="prediction-label-nodata">{data}</span>
      </li>
    );
  }

  // Caso 3: 'data' es el objeto completo con la predicción y las métricas.
  // Generamos una clase CSS dinámica para poder darle color a la etiqueta.
  const labelClass = data.prediccion ? data.prediccion.toLowerCase().split('/')[0].replace(' ', '-') : 'nodata';

  return (
    <li className="prediction-item">
      <div className="prediction-main">
        <strong>{targetName}:</strong>
        <span className={`prediction-label prediction-${labelClass}`}>
          {data.prediccion || 'N/D'}
        </span>
      </div>
      <div className="metrics">
        <span><strong>Modelo:</strong> {data.modelo_usado || 'N/D'}</span>
        <span><strong>F1 (CV):</strong> {data.f1_score_cv || 'N/D'}</span>
        <span><strong>ROC AUC (CV):</strong> {data.roc_auc_cv || 'N/D'}</span>
      </div>
    </li>
  );
}


// Componente principal para seleccionar opciones y predecir
function Predict() {
  const [selectedOption, setSelectedOption] = useState('');
  const [predictionResult, setPredictionResult] = useState([]);
  const [options, setOptions] = useState([]);
  const [activeTab, setActiveTab] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // Para mostrar un feedback de carga
  const navigate = useNavigate();

  // Carga las opciones del dropdown una sola vez cuando el componente se monta
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await axios.get('http://localhost:5001/get-options');
        setOptions(['Todos', ...response.data]);
      } catch (error) {
        console.error('Error al obtener las opciones:', error);
      }
    };
    fetchOptions();
  }, []);

  // Función para manejar la llamada a la API de predicción
  const handlePredict = async () => {
    if (!selectedOption) {
      alert('Por favor, selecciona una opción antes de predecir.');
      return;
    }
    
    // Evitar predicciones duplicadas en las pestañas
    if (selectedOption !== 'Todos' && predictionResult.some(result => result.option === selectedOption)) {
      alert('Ya existe una predicción para este sitio. Ciérrala para volver a predecir.');
      setActiveTab(selectedOption);
      return;
    }

    setIsLoading(true); // Activa el estado de carga
    try {
      const response = await axios.post('http://localhost:5001/predict', {
        option: selectedOption,
      });

      // La API siempre devuelve un array, lo procesamos
      const newPredictions = response.data.map(prediction => ({
        ...prediction,
        option: prediction.codigo_perfil
      }));
      
      // Si se predijo "Todos", reemplazamos los resultados. Si no, los añadimos.
      if (selectedOption === 'Todos') {
        setPredictionResult(newPredictions);
        setActiveTab(newPredictions.length > 0 ? newPredictions[0].option : null);
      } else {
        setPredictionResult(prevResults => [...prevResults, ...newPredictions]);
        setActiveTab(newPredictions[0].option);
      }

    } catch (error) {
      console.error('Error al obtener la predicción:', error);
      alert('Ocurrió un error al contactar el servidor de predicción.');
    } finally {
      setIsLoading(false); // Desactiva el estado de carga
    }
  };

  const handleRemovePrediction = (optionToRemove) => {
    setPredictionResult(prevResults => {
      const updatedResults = prevResults.filter(result => result.option !== optionToRemove);
      // Si la pestaña cerrada era la activa, activamos la primera de la lista o ninguna
      if (activeTab === optionToRemove) {
        setActiveTab(updatedResults.length > 0 ? updatedResults[0].option : null);
      }
      return updatedResults;
    });
  };

  // Encontramos el objeto del resultado activo para no repetir la búsqueda en el JSX
  const activeResult = activeTab ? predictionResult.find(result => result.option === activeTab) : null;

  return (
    <div className="container">
      <div className="titulo">
      <h2>Realizar Predicción</h2>
      </div>
      
      <div className="predict-controls">
        <select
          value={selectedOption}
          onChange={(e) => setSelectedOption(e.target.value)}
          disabled={isLoading}
        >
          <option value="">Selecciona un sitio</option>
          {options.map((option) => (
            <option key={option} value={option}>{option}</option>
          ))}
        </select>
        <button onClick={handlePredict} className="primary-button" disabled={isLoading}>
          {isLoading ? 'Prediciendo...' : 'Predecir'}
        </button>
      </div>

      {predictionResult.length > 0 && (
        <div className="prediction-container">
          <div className="tabs-header">
            {predictionResult.map((result) => (
              <div
                key={result.option}
                className={`tab-title ${activeTab === result.option ? 'active' : ''}`}
                onClick={() => setActiveTab(result.option)}
              >
                {`Sitio: ${result.option}`}
                <button 
                  className="close-tab" 
                  onClick={(e) => { e.stopPropagation(); handleRemovePrediction(result.option); }}
                  title={`Cerrar predicción para ${result.option}`}
                >×</button>
              </div>
            ))}
          </div>
          
          {activeResult && (
            <div className="prediction-tab">
              <h3>Resultados para {activeResult.option}</h3>
              <ul>
                <PredictionDetails targetName="Clorofila" data={activeResult.Clorofila} />
                <PredictionDetails targetName="Cianobacterias" data={activeResult.Cianobacterias} />
                <PredictionDetails targetName="Dominancia" data={activeResult.Dominancia} />
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// Componente para mostrar los datos del DataFrame
function Datos() {
  const [tableData, setTableData] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [isUpdating, setIsUpdating] = useState(false); // <-- 1. NUEVO ESTADO
  const itemsPerPage = 4;
  const navigate = useNavigate();

  // La función para obtener los datos ahora se puede reutilizar
  const fetchData = async () => {
    try {
      const response = await axios.get('http://localhost:5001/datos');
      setTableData(response.data);
    } catch (error) {
      console.error('Error al obtener los datos:', error);
      // Opcional: limpiar la tabla si hay un error al cargar
      setTableData([]);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);


  // --- 2. NUEVA FUNCIÓN PARA LLAMAR A /actualizar ---
  const handleUpdate = async () => {
    setIsUpdating(true); // Deshabilitar botón
    alert('Iniciando actualización de datos en el servidor. Este proceso puede tardar varios minutos. Por favor, espera a que aparezca el mensaje de confirmación.');

    try {
      const response = await axios.post('http://localhost:5001/actualizar');
      // Cuando el proceso termina, muestra el mensaje de éxito
      alert(response.data.message);
      // Vuelve a cargar los datos en la tabla para ver los cambios
      fetchData(); 
    } catch (error) {
      console.error('Error al actualizar los datos:', error);
      const errorMessage = error.response ? error.response.data.error : 'Error de conexión.';
      alert(`Error al actualizar: ${errorMessage}`);
    } finally {
      setIsUpdating(false); // Vuelve a habilitar el botón
    }
  };
  // --- FIN DE LA NUEVA FUNCIÓN ---


  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentItems = tableData.slice(startIndex, startIndex + itemsPerPage);
  const totalPages = Math.ceil(tableData.length / itemsPerPage);

  // Crear el rango de botones a mostrar
  const getPaginationButtons = () => {
    const buttons = [];
    const maxButtons = 5; // Cantidad máxima de botones que se muestran

    let startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
    let endPage = Math.min(totalPages, startPage + maxButtons - 1);

    if (endPage - startPage < maxButtons - 1) {
      startPage = Math.max(1, endPage - maxButtons + 1);
    }

    for (let i = startPage; i <= endPage; i++) {
      buttons.push(
        <button
          key={i}
          onClick={() => handlePageChange(i)}
          className={currentPage === i ? 'active' : ''}
        >
          {i}
        </button>
      );
    }
    return buttons;
  };

  return (
    <div className="container">
      <button onClick={() => navigate('/')} className="back-button">&larr;</button>
      <img src={logo} alt="Logo" className="logo2" />
      <h2>Datos del DataFrame</h2>

      {/* --- 3. NUEVO BOTÓN DE ACTUALIZAR --- */}
      <div style={{ margin: '20px 0' }}>
        <button onClick={handleUpdate} disabled={isUpdating} className="primary-button">
          {isUpdating ? 'Procesando en Servidor...' : 'Forzar Actualización de Datos'}
        </button>
      </div>
      {/* --- FIN DEL NUEVO BOTÓN --- */}

      {tableData.length > 0 ? (
        <>
          <div className="table-container">
            <table border="1">
              <thead>
                <tr>
                  {Object.keys(tableData[0]).map((key) => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {currentItems.map((row, index) => (
                  <tr key={index}>
                    {Object.values(row).map((value, i) => (
                      // Añadido un pequeño cambio para que los 'null' no se muestren
                      <td key={i}>{value === null ? "" : String(value)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pagination">
            <button onClick={() => handlePageChange(currentPage - 1)} disabled={currentPage === 1}>
              &laquo; Anterior
            </button>
            {getPaginationButtons()}
            <button onClick={() => handlePageChange(currentPage + 1)} disabled={currentPage === totalPages}>
              Siguiente &raquo;
            </button>
          </div>
        </>
      ) : (
        <p>Cargando datos...</p>
      )}
    </div>
  );
}

// Componente para mostrar los datos del DataFrame
function Predicciones() {
  const [tableData, setTableData] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 4;
  const navigate = useNavigate();

  // La función para obtener los datos ahora se puede reutilizar
  const fetchData = async () => {
    try {
      const response = await axios.get('http://localhost:5001/predicciones');
      setTableData(response.data);
    } catch (error) {
      console.error('Error al obtener los datos:', error);
      // Opcional: limpiar la tabla si hay un error al cargar
      setTableData([]);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentItems = tableData.slice(startIndex, startIndex + itemsPerPage);
  const totalPages = Math.ceil(tableData.length / itemsPerPage);

  // Crear el rango de botones a mostrar
  const getPaginationButtons = () => {
    const buttons = [];
    const maxButtons = 10; // Cantidad máxima de botones que se muestran

    let startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
    let endPage = Math.min(totalPages, startPage + maxButtons - 1);

    if (endPage - startPage < maxButtons - 1) {
      startPage = Math.max(1, endPage - maxButtons + 1);
    }

    for (let i = startPage; i <= endPage; i++) {
      buttons.push(
        <button
          key={i}
          onClick={() => handlePageChange(i)}
          className={currentPage === i ? 'active' : ''}
        >
          {i}
        </button>
      );
    }
    return buttons;
  };

  return (
    <div className="container">
      <button onClick={() => navigate('/')} className="back-button">&larr;</button>
      <img src={logo} alt="Logo" className="logo2" />
      <h2>Datos del DataFrame</h2>

      {tableData.length > 0 ? (
        <>
          <div className="table-container">
            <table border="1">
              <thead>
                <tr>
                  {Object.keys(tableData[0]).map((key) => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {currentItems.map((row, index) => (
                  <tr key={index}>
                    {Object.values(row).map((value, i) => (
                      // Añadido un pequeño cambio para que los 'null' no se muestren
                      <td key={i}>{value === null ? "" : String(value)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pagination">
            <button onClick={() => handlePageChange(currentPage - 1)} disabled={currentPage === 1}>
              &laquo; Anterior
            </button>
            {getPaginationButtons()}
            <button onClick={() => handlePageChange(currentPage + 1)} disabled={currentPage === totalPages}>
              Siguiente &raquo;
            </button>
          </div>
        </>
      ) : (
        <p>Cargando datos...</p>
      )}
    </div>
  );
}

// Componente de la aplicación principal
function App() {
  const navigate = useNavigate();

  const handlePredictStart = () => {
    navigate('/predict');
  };

  const handleDataStart = () => {
    navigate('/datos');
  };

  const handlePredictionsStart = () => {
    navigate('/predicciones');
  };

  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Home onPredictStart={handlePredictStart} onDataStart={handleDataStart} onPredictionsStart={handlePredictionsStart} />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/datos" element={<Datos />} />
        <Route path="/predicciones" element={<Predicciones />} />
      </Routes>
    </div>
  );
}

// Componente raíz con Router
function AppRoot() {
  return (
    <Router>
      <App />
    </Router>
  );
}

export default AppRoot;

// Notas:
// 1. Asegúrate de que el backend esté corriendo en http://localhost:5001 o cambia la URL según corresponda.
// 2. Este código asume que el backend tiene un endpoint POST en '/predict' que maneja la lógica de predicción.