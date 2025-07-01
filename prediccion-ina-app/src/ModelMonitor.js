import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useNavigate } from 'react-router-dom';

function ModelMonitor() {
    const [historyData, setHistoryData] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const navigate = useNavigate();

    // Estados para los filtros
    const [selectedSitio, setSelectedSitio] = useState('');
    const [selectedTarget, setSelectedTarget] = useState('');

    // Cargar datos históricos al montar el componente
    useEffect(() => {
        const fetchData = async () => {
            setIsLoading(true);
            try {
                const response = await axios.get('/api/metricas-historicas');
                if (response.data.length > 0) {
                    setSelectedSitio(response.data[0].sitio);
                    setSelectedTarget(response.data[0].variable_objetivo);
                }
                setHistoryData(response.data);
            } catch (error) {
                console.error('Error al cargar el historial de métricas:', error);
            } finally {
                setIsLoading(false);
            }
        };
        fetchData();
    }, []);

    // Opciones para los menús desplegables (calculadas a partir de los datos)
    const sitioOptions = useMemo(() => [...new Set(historyData.map(item => item.sitio))], [historyData]);
    const targetOptions = useMemo(() => [...new Set(historyData.map(item => item.variable_objetivo))], [historyData]);

    // Filtrar los datos para el gráfico basado en la selección del usuario
    const chartData = useMemo(() => {
        return historyData
            .filter(item => item.sitio === selectedSitio && item.variable_objetivo === selectedTarget)
            .map(item => ({
                ...item,
                fecha: new Date(item.timestamp_entrenamiento).toLocaleDateString('es-AR')
            }));
    }, [historyData, selectedSitio, selectedTarget]);

    if (isLoading) {
        return <div className="container"><p>Cargando datos de monitorización...</p></div>;
    }

    return (
        <div className="container">
            <button onClick={() => navigate('/')} className="back-button">&larr;</button>
            <h2>Evolución del Rendimiento de Modelos</h2>
            
            <div className="filters-container" style={{ margin: '20px 0' }}>
                <select value={selectedSitio} onChange={(e) => setSelectedSitio(e.target.value)}>
                    <option value="">Seleccione un Sitio</option>
                    {sitioOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                </select>
                <select value={selectedTarget} onChange={(e) => setSelectedTarget(e.target.value)}>
                    <option value="">Seleccione una Variable</option>
                    {targetOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                </select>
            </div>

            {chartData.length > 0 ? (
                <div style={{ width: '100%', height: 400 }}>
                    <ResponsiveContainer>
                        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="fecha" />
                            <YAxis domain={[0, 1]} />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="f1_score_cv" name="F1-Score (CV)" stroke="#8884d8" activeDot={{ r: 8 }} />
                            <Line type="monotone" dataKey="roc_auc_cv" name="ROC AUC (CV)" stroke="#82ca9d" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            ) : (
                <p>No hay datos de entrenamiento para la selección actual.</p>
            )}
        </div>
    );
}

export default ModelMonitor;
