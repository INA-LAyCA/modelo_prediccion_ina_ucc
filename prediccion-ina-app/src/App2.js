// Dependencias necesarias
// En el directorio del proyecto, instalar las siguientes dependencias:
// npm install axios react-router-dom

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import './App.css';
import logo from './logo.png';

// Componente de pantalla de inicio
function Home({ onPredictStart, onDataStart }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <img src={logo} alt="Logo" className="logo1" />
      <h1>MODELO DE PREDICCIÓN - INA CIRSA</h1>
      <div style={{ display: 'flex', gap: '20px', marginTop: '20px' }}>
        <button onClick={onPredictStart} className="primary-button">Predecir</button>
        <button onClick={onDataStart} className="primary-button">Ver Datos</button>
      </div>
    </div>
  );
}
// Componente principal para seleccionar opciones y predecir
function Predict() {
  const [selectedOption, setSelectedOption] = useState('');
  const [predictionResult, setPredictionResult] = useState([]);
  const [options, setOptions] = useState([]);
  const [activeTab, setActiveTab] = useState(null);
  const navigate = useNavigate(); // para usar la navegación

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

  const handlePredict = async () => {
    if (selectedOption) {
      // Verificar si ya existe una predicción para el sitio seleccionado
      if (predictionResult.some(result => result.option === selectedOption)) {
        alert('Ya existe una predicción para este sitio.');
        return;
      }
      try {
        const response = await axios.post('http://localhost:5001/predict', {
          option: selectedOption,
        });
        if (selectedOption === 'Todos') {
          setPredictionResult((prevResults) => [...prevResults, ...response.data.map(prediction => ({ ...prediction, option: prediction.codigo_perfil }))]);
        } else {
          setPredictionResult((prevResults) => [...prevResults, { ...response.data, option: selectedOption }]);
        }
        setActiveTab(selectedOption); // Activar el tab de la nueva predicción
      } catch (error) {
        console.error('Error al obtener la predicción:', error);
      }
    } else {
      alert('Por favor, selecciona una opción antes de predecir.');
    }
  };

  const handleRemovePrediction = (option) => {
    setPredictionResult((prevResults) => {
      const updatedResults = prevResults.filter((result) => result.option !== option);
      if (activeTab === option) {
        setActiveTab(updatedResults.length > 0 ? updatedResults[0].option : null);
      }
      return updatedResults;
    });
  };

  const handleTabClick = (option) => {
    setActiveTab(option);
  };

  return (
    <div className="container">
      <button onClick={() => navigate('/')} className="back-button">&larr; Volver</button>
      <img src={logo} alt="Logo" className="logo2" />
      <h2>Realizar Predicción</h2>
      <select
        value={selectedOption}
        onChange={(e) => setSelectedOption(e.target.value)}
        style={{ width: '50%', fontSize: '20px', marginTop: '20px' }}
      >
        <option value="">Selecciona una opción</option>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
      <button onClick={handlePredict} className="primary-button" style={{ marginTop: '20px' }}>Predecir</button>

      {predictionResult.length > 0 && (
        <div className="prediction-container">
          <div className="tabs-header">
            {predictionResult.map((result) => (
              <div
                key={result.option}
                className={`tab-title ${activeTab === result.option ? 'active' : ''}`}
                onClick={() => handleTabClick(result.option)}
              >
                {`Sitio seleccionado: ${result.option}`}
                <button className="close-tab" onClick={() => handleRemovePrediction(result.option)}>×</button>
              </div>
            ))}
          </div>
          {activeTab && predictionResult.some(result => result.option === activeTab) && (
            <div className="prediction-tab">
              <p>Predicción:</p>
              <ul>
                <li>Cianobacterias Total: {predictionResult.find(result => result.option === activeTab)['Cianobacterias Total']}</li>
                <li>Clorofila (µg/l): {predictionResult.find(result => result.option === activeTab)['Clorofila (µg/l)']}</li>
                <li>Dominancia de Cianobacterias (%): {predictionResult.find(result => result.option === activeTab)['Dominancia de Cianobacterias (%)']}</li>
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
  const itemsPerPage = 4;
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5001/datos');
        setTableData(response.data);
      } catch (error) {
        console.error('Error al obtener los datos:', error);
      }
    };
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
      <button onClick={() => navigate('/')} className="back-button">&larr; Volver</button>
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
                      <td key={i}>{value}</td>
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

  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Home onPredictStart={handlePredictStart} onDataStart={handleDataStart} />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/datos" element={<Datos />} />
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
