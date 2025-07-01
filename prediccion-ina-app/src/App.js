import React, { useState, useEffect, useMemo } from 'react';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './App.css';
import logo from './logo.png';
import AboutPage from './AboutPage';
import ModelMonitor from './ModelMonitor';



// Componente de pantalla de inicio
function Home({ onPredictStart, onDataStart, onPredictionsStart, onAboutStart, onMonitorStart }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <img src={logo} alt="Logo" className="logo1" />
      <h1>MODELO DE PREDICCIÓN - INA CIRSA</h1>
      <div style={{ display: 'flex', gap: '20px', marginTop: '20px' }}>
        <button onClick={onPredictStart} className="primary-button">Predecir</button>
        <button onClick={onDataStart} className="primary-button">Ver Datos</button>
        <button onClick={onPredictionsStart} className="primary-button">Ver Predicciones</button>
        <button onClick={onAboutStart} className="primary-button">Acerca del Modelo</button>
        <button onClick={onMonitorStart} className="primary-button">Monitorear Modelos</button>
        
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
export function Predict() {
  const [selectedOption, setSelectedOption] = useState('');
  const [predictionResult, setPredictionResult] = useState([]);
  const [options, setOptions] = useState([]);
  const [activeTab, setActiveTab] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // feedback de carga
  const navigate = useNavigate();

  // Carga las opciones del desplegable
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await axios.get('/api/get-options');
        setOptions(['Todos', ...response.data]);
      } catch (error) {
        console.error('Error al obtener las opciones:', error);
      }
    };
    fetchOptions();
  }, []);

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
      const response = await axios.post('/api/predict', {
        option: selectedOption,
      });

      const newPredictions = response.data.map(prediction => ({
        ...prediction,
        option: prediction.codigo_perfil
      }));
      
      // Si se predijo "Todos", reemplazamos los resultados, si no, los agregamos
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

  const formatPredictionDate = (dateString) => {
    if (!dateString) return '';
    try {
      const date = new Date(dateString);
      // 'es-ES' para nombres de meses en español. 'long' para el nombre completo.
      const month = date.toLocaleString('es-ES', { month: 'long' });
      const year = date.getFullYear();
      // Capitaliza la primera letra del mes
      return `${month.charAt(0).toUpperCase() + month.slice(1)} ${year}`;
    } catch (error) {
      return '';
    }
  };

  return (
    <div className="container">
      <button onClick={() => navigate('/')} className="back-button">&larr;</button>
      <img src={logo} alt="Logo" className="logo2" />
      <h2>Realizar Predicción</h2>
      
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
              <h3>Resultados para {activeResult.option} en <strong>{formatPredictionDate(activeResult.fecha_prediccion)}</strong></h3>
              <ul>
                <PredictionDetails targetName="Clorofila" data={activeResult.Clorofila} />
                <PredictionDetails targetName="Cianobacterias" data={activeResult.Cianobacterias} />
                <PredictionDetails targetName="Dominancia de Cianobacterias" data={activeResult.Dominancia} />
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}



function Datos() {
  const [tableData, setTableData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();
  
  const [filterSitio, setFilterSitio] = useState('');
  const [filterAnio, setFilterAnio] = useState('');
  const [filterMes, setFilterMes] = useState('');
  const [filterEstacion, setFilterEstacion] = useState('');
  
  const [sitioOptions, setSitioOptions] = useState([]);
  const [anioOptions, setAnioOptions] = useState([]);
  const [mesOptions, setMesOptions] = useState([]);
  const [estacionOptions, setEstacionOptions] = useState([]);

  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 18;

  const COLUMNAS_DATOS = useMemo(() => [
    { accessor: 'id_registro', Header: 'Id Registro' },
    { accessor: 'fecha', Header: 'Fecha' },
    { accessor: 'codigo_perfil', Header: 'Sitio' },
    { accessor: 'estacion', Header: 'Estación' },
    { accessor: 'Clorofila (µg/l)', Header: 'Clorofila (µg/L)' },
    { accessor: 'Cianobacterias Total', Header: 'Cianobacterias (cel/L)' },
    { accessor: 'Dominancia de Cianobacterias (%)', Header: 'Dominancia Ciano (%)' },
    { accessor: 'Nitrogeno Inorganico Total (µg/l)', Header: 'Nitrógeno Total (µg/L)' },
    { accessor: 'T° (°C)', Header: 'Temp. Agua (°C)' },
    { accessor: 'condicion_termica', Header: 'Condición Térmica' },
    { accessor: 'Cota (m)', Header: 'Cota (m)' },
    { accessor: 'PHT (µg/l)', Header: 'PHT (µg/l)' },
    { accessor: 'PRS (µg/l)', Header: 'PRS (µg/l)' },
    { accessor: 'temperatura_min', Header: 'Temperatura Minima' },
    { accessor: 'temperatura_max', Header: 'Temperatura Maxima' },
    { accessor: '600', Header: '600 Bo El Canal' },
    { accessor: '700', Header: '700 Confluencia el Cajon' },
    { accessor: '1100', Header: '1100 CIRSA Villa Carlos Paz' }
    
  ], []);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const response = await axios.get('/api/datos');
        const data = Array.isArray(response.data) ? response.data : [];
        setTableData(data);

        if (data.length > 0) {
          setSitioOptions([...new Set(data.map(item => item.codigo_perfil).filter(Boolean))].sort());
          setMesOptions([...new Set(data.map(item => new Date(item.fecha).getMonth()+ 1))].sort((a, b) => b - a));
          setAnioOptions([...new Set(data.map(item => new Date(item.fecha).getFullYear()))].sort((a, b) => b - a));
          setEstacionOptions([...new Set(data.map(item => item.estacion).filter(Boolean))].sort());
        }
      } catch (error) {
        console.error('Error al obtener los datos:', error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  const filteredData = useMemo(() => {
    return tableData.filter(row => {
      const matchSitio = filterSitio ? row.codigo_perfil === filterSitio : true;
      const matchAnio = filterAnio ? new Date(row.fecha).getFullYear() === parseInt(filterAnio) : true;
      const matchMes = filterMes ? new Date(row.fecha).getMonth() === parseInt(filterMes) : true;
      const matchEstacion = filterEstacion ? row.estacion === filterEstacion : true;
      return matchSitio && matchAnio && matchMes && matchEstacion;
    });
  }, [tableData, filterSitio, filterAnio, filterMes, filterEstacion]);

  useEffect(() => {
    setCurrentPage(1);
  }, [filteredData]);

  const totalPages = Math.ceil(filteredData.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentItems = filteredData.slice(startIndex, startIndex + itemsPerPage);

  const getPaginationButtons = () => {
    const buttons = [];
    const maxButtons = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
    let endPage = Math.min(totalPages, startPage + maxButtons - 1);
    if (endPage - startPage < maxButtons - 1) {
      startPage = Math.max(1, endPage - maxButtons + 1);
    }
    for (let i = startPage; i <= endPage; i++) {
      buttons.push(
        <button key={i} onClick={() => setCurrentPage(i)} className={currentPage === i ? 'active' : ''}>{i}</button>
      );
    }
    return buttons;
  };

  return (
    <div className="container">
      <div className="page-header">
        <button onClick={() => navigate('/')} className="back-button">&larr;</button>
        <h2>Tabla de Datos Procesados</h2>
      </div>
      
      <div className="filters-container">
        <select className="filter-select" value={filterSitio} onChange={(e) => setFilterSitio(e.target.value)}>
          <option value="">Todos los Sitios</option>
          {sitioOptions.map(sitio => <option key={sitio} value={sitio}>{sitio}</option>)}
        </select>
        <select className="filter-select" value={filterAnio} onChange={(e) => setFilterAnio(e.target.value)}>
          <option value="">Todos los Años</option>
          {anioOptions.map(anio => <option key={anio} value={anio}>{anio}</option>)}
        </select>
        <select className="filter-select" value={filterMes} onChange={(e) => setFilterMes(e.target.value)}>
          <option value="">Todos los Meses</option>
          {mesOptions.map(mes => <option key={mes} value={mes}>{mes}</option>)}
        </select>
        <select className="filter-select" value={filterEstacion} onChange={(e) => setFilterEstacion(e.target.value)}>
          <option value="">Todas las Estaciones</option>
          {estacionOptions.map(estacion => <option key={estacion} value={estacion}>{estacion}</option>)}
        </select>
      </div>

      {isLoading ? (<p>Cargando datos...</p>) : (
        <>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                 {COLUMNAS_DATOS.map(columna => (
                    <th key={columna.accessor}>{columna.Header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
              {currentItems.map((row, index) => (
                <tr key={index}>
                  {COLUMNAS_DATOS.map(columna => (
                    <td key={`${index}-${columna.accessor}`}>
                      {columna.accessor === 'fecha'
                        ? formatDate(row[columna.accessor])
                        : (row[columna.accessor] === null ? "" : String(row[columna.accessor]))
                      }
                    </td>
                  ))}
                </tr>
              ))}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && <div className="pagination">
            <button onClick={() => setCurrentPage(c => Math.max(1, c - 1))} disabled={currentPage === 1}>&laquo; Anterior</button>
            {getPaginationButtons()}
            <button onClick={() => setCurrentPage(c => Math.min(totalPages, c + 1))} disabled={currentPage === totalPages}>Siguiente &raquo;</button>
          </div>}
        </>
      )}
    </div>
  );
}

// Componente para mostrar los datos del DataFrame
function Predicciones() {
  const [tableData, setTableData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();
  
  const [filterSitio, setFilterSitio] = useState('');
  const [filterTarget, setFilterTarget] = useState('');
  const [filterAlerta, setFilterAlerta] = useState('');
  const [filterAnio, setFilterAnio] = useState('');
  const [filterMes, setFilterMes] = useState('');
  
  const [sitioOptions, setSitioOptions] = useState([]);
  const [targetOptions, setTargetOptions] = useState([]);
  const [alertaOptions, setAlertaOptions] = useState([]);
  const [anioOptions, setAnioOptions] = useState([]);
  const [mesOptions, setMesOptions] = useState([]);
  
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 18;

  const COLUMNAS_PREDICCIONES = useMemo(() => [
    { accessor: 'id_prediccion', Header: 'ID Prediccion' },
    { accessor: 'timestamp_ejecucion', Header: 'Fecha de Ejecución' },
    { accessor: 'fecha_prediccion', Header: 'Fecha de Predicción' },
    { accessor: 'codigo_perfil', Header: 'Sitio' },
    { accessor: 'target', Header: 'Variable' },
    { accessor: 'etiqueta_predicha', Header: 'Etiqueta' }
], []);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const response = await axios.get('/api/predicciones');
        const data = Array.isArray(response.data) ? response.data : [];
        setTableData(data);

        if (data.length > 0) {
          setSitioOptions([...new Set(data.map(item => item.codigo_perfil).filter(Boolean))].sort());
          setTargetOptions([...new Set(data.map(item => item.target).filter(Boolean))].sort());
          setAlertaOptions([...new Set(data.map(item => item.etiqueta_predicha).filter(Boolean))].sort());
          const anios = [...new Set(data.map(item => new Date(item.fecha_prediccion).getFullYear()))].sort((a, b) => b - a);
          setAnioOptions(anios);
          
          const meses = [...new Set(data.map(item => {
              const monthIndex = new Date(item.fecha_prediccion).getMonth();
              return new Date(0, monthIndex).toLocaleString('es-ES', { month: 'long' });
          }))];
          setMesOptions(meses);
        }
      } catch (error) {
        console.error('Error al obtener los datos:', error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  const filteredData = useMemo(() => {
    return tableData.filter(row => {
      const matchSitio = filterSitio ? row.codigo_perfil === filterSitio : true;
      const matchTarget = filterTarget ? row.target === filterTarget : true;
      const matchAlerta = filterAlerta ? row.etiqueta_alerta === filterAlerta : true;
      const matchAnio = filterAnio ? new Date(row.fecha_prediccion).getFullYear() === parseInt(filterAnio) : true;
      
      const monthName = new Date(row.fecha_prediccion).toLocaleString('es-ES', { month: 'long' });
      const matchMes = filterMes ? monthName.toLowerCase() === filterMes.toLowerCase() : true;

      return matchSitio && matchTarget && matchAlerta && matchAnio && matchMes;
    });
  }, [tableData, filterSitio, filterTarget, filterAlerta, filterAnio, filterMes]);

  useEffect(() => {
    setCurrentPage(1);
  }, [filteredData]);

  const totalPages = Math.ceil(filteredData.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentItems = filteredData.slice(startIndex, startIndex + itemsPerPage);

  const getPaginationButtons = () => {
    const buttons = [];
    const maxButtons = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
    let endPage = Math.min(totalPages, startPage + maxButtons - 1);
    if (endPage - startPage < maxButtons - 1) {
      startPage = Math.max(1, endPage - maxButtons + 1);
    }
    for (let i = startPage; i <= endPage; i++) {
      buttons.push(
        <button key={i} onClick={() => setCurrentPage(i)} className={currentPage === i ? 'active' : ''}>{i}</button>
      );
    }
    return buttons;
  };

  return (
    <div className="container">
      <div className="page-header">
        <button onClick={() => navigate('/')} className="back-button">&larr;</button>
        <h2>Historial de predicciones</h2>
      </div>
      
      <div className="filters-container">
          <select className="filter-select" value={filterSitio} onChange={(e) => setFilterSitio(e.target.value)}>
              <option value="">Todos los Sitios</option>
              {sitioOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <select className="filter-select" value={filterTarget} onChange={(e) => setFilterTarget(e.target.value)}>
              <option value="">Todos los Targets</option>
              {targetOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <select className="filter-select" value={filterAlerta} onChange={(e) => setFilterAlerta(e.target.value)}>
              <option value="">Todas las Alertas</option>
              {alertaOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <select className="filter-select" value={filterAnio} onChange={(e) => setFilterAnio(e.target.value)}>
              <option value="">Todos los Años</option>
              {anioOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <select className="filter-select" value={filterMes} onChange={(e) => setFilterMes(e.target.value)}>
              <option value="">Todos los Meses</option>
              {mesOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
          </select>
      </div>

      {isLoading ? (<p>Cargando datos...</p>) : (
        <>
          <div className="table-container">
            <table>
              <thead>
                 <tr>
                    {COLUMNAS_PREDICCIONES.map(col => <th key={col.accessor}>{col.Header}</th>)}
                  </tr>
              </thead>
              <tbody>
                {currentItems.map((row, index) => (
                  <tr key={index}>
                      {COLUMNAS_PREDICCIONES.map(col => (
                          <td key={`${index}-${col.accessor}`}>
                            {(col.accessor === 'timestamp_ejecucion' || col.accessor === 'fecha_prediccion')
                              ? formatDate(row[col.accessor])
                              : (row[col.accessor] === null ? "" : String(row[col.accessor]))
                            }
                          </td>
                      ))}
                  </tr>
                  ))}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && <div className="pagination">
            <button onClick={() => setCurrentPage(c => Math.max(1, c - 1))} disabled={currentPage === 1}>&laquo; Anterior</button>
            {getPaginationButtons()}
            <button onClick={() => setCurrentPage(c => Math.min(totalPages, c + 1))} disabled={currentPage === totalPages}>Siguiente &raquo;</button>
          </div>}
        </>
      )}
    </div>
  );
}

export function formatDate(dateString) {
  if (!dateString) {
    return '-';
  }

  const date = new Date(dateString);

  // Verifica si el objeto de fecha es válido.
  if (isNaN(date.getTime())) {
    return 'Fecha inválida';
  }

  // Para evitar problemas de zona horaria (timezone), en lugar de .getDate(), .getMonth(), etc.,
  // usamos los métodos UTC que siempre se refieren al Tiempo Universal Coordinado.
  // Esto previene que la fecha cambie al día anterior/posterior.
  const day = String(date.getUTCDate()).padStart(2, '0');
  const month = String(date.getUTCMonth() + 1).padStart(2, '0'); // getUTCMonth es 0-11
  const year = date.getUTCFullYear();

  return `${day}/${month}/${year}`;
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

  const handleAboutStart = () => {
    navigate('/about');
  };
  const handleMonitorStart = () => navigate('/monitor');

  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Home onPredictStart={handlePredictStart} onDataStart={handleDataStart} onPredictionsStart={handlePredictionsStart} onAboutStart={handleAboutStart} onMonitorStart={handleMonitorStart}/>} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/datos" element={<Datos />} />
        <Route path="/predicciones" element={<Predicciones />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/monitor" element={<ModelMonitor />} />
      </Routes>
    </div>
  );
}



function RetrainingOverlay() {
  const overlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    color: 'white',
    fontSize: '24px',
    zIndex: 9999,
    fontFamily: 'Arial, sans-serif',
  };

  return (
    <div style={overlayStyle}>
      <p>Espere, los modelos están siendo actualizados y reentrenados...</p>
    </div>
  );
}


// Componente raíz con Router
function AppRoot() {
  const [isRetraining, setIsRetraining] = useState(false);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await axios.get('/api/status');
        const isCurrentlyRetraining = response.data.status === 'retraining';
        
        setIsRetraining(isCurrentlyRetraining);
      } catch (error) {
        console.error("Error al consultar el estado del backend:", error);
        setIsRetraining(true);
      }
    };

    checkStatus();
    const intervalId = setInterval(checkStatus, 3000);

    return () => {
      clearInterval(intervalId);
    };
  },[]);

  return (
    <Router>
      {isRetraining && <RetrainingOverlay />}
      <App />
    </Router>
  );
}
export { Home, PredictionDetails};
export default AppRoot;
