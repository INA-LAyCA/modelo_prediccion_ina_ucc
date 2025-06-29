// frontend/src/App.test.js

import React from 'react';
import { render, screen, fireEvent, waitFor, within  } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';
import { MemoryRouter } from 'react-router-dom';

// Importamos los componentes NOMBRADOS directamente desde App.js
// y también la función de utilidad.
import { Predict, formatDate, PredictionDetails } from './App';

// Mockear axios para controlar las respuestas de la API
jest.mock('axios');

// --- Prueba Unitaria para una función de utilidad ---
describe('formatDate utility function', () => {
  test('should format YYYY-MM-DD to DD/MM/YYYY', () => {
    // Corregí la descripción para que coincida con tu función
    expect(formatDate('2025-07-28')).toBe('28/07/2025');
  });

  test('should return "Fecha inválida" for incorrect formats', () => {
    expect(formatDate('28-07-2025')).toBe('Fecha inválida');
  });

  test('should return "-" for null or undefined input', () => {
    expect(formatDate(null)).toBe('-');
  });
});


// --- Prueba de Integración para el componente Predict ---
describe('Predict Component', () => {
  test('fetches options and makes a prediction on button click', async () => {
    // 1. Definir las respuestas simuladas de la API
    const mockOptions = ['C1', 'TAC1', 'DSA1'];
    const mockPrediction = [{
      codigo_perfil: 'C1',
      fecha_prediccion: '2025-07-01',
      Clorofila: { prediccion: 'Alerta', modelo_usado: 'RandomForest', f1_score_cv: 0.8, roc_auc_cv: 0.9 },
      Cianobacterias: { prediccion: 'Vigilancia', modelo_usado: 'MLP', f1_score_cv: 0.85, roc_auc_cv: 0.92 },
      Dominancia: { prediccion: 'No Dominante', modelo_usado: 'LogisticRegression', f1_score_cv: 0.9, roc_auc_cv: 0.95 }
    }];

    // Configurar el mock de axios para las llamadas esperadas
    axios.get.mockResolvedValue({ data: mockOptions });
    axios.post.mockResolvedValue({ data: mockPrediction });

    // 2. Renderizar el componente
    render(
      <MemoryRouter>
        <Predict />
      </MemoryRouter>
    );

    // 3. Verificar que las opciones se cargaron
    await waitFor(() => {
      expect(screen.getByText('C1')).toBeInTheDocument();
    });

    // 4. Simular la interacción del usuario
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'C1' } });
    fireEvent.click(screen.getByRole('button', { name: /predecir/i }));

    // 5. Verificar que los resultados de la predicción se muestren en pantalla
   // ESTA ES LA VERSIÓN CORREGIDA
    await waitFor(() => {
      // Verificamos que los resultados generales están visibles
      expect(screen.getByText(/Resultados para C1/i)).toBeInTheDocument();
      expect(screen.getByText('Alerta')).toBeInTheDocument();
    });

    // Ahora, verificamos las métricas de una forma más robusta
    // 1. Encontramos el elemento <li> que contiene la palabra "Clorofila"
    const clorofilaItem = screen.getByText(/Clorofila/i).closest('li');

    // 2. Usamos `within` para buscar texto solo DENTRO de ese elemento <li>
    // Esto asegura que estamos viendo las métricas del modelo correcto
    expect(within(clorofilaItem).getByText(/RandomForest/i)).toBeInTheDocument();
    expect(within(clorofilaItem).getByText(/0.8/)).toBeInTheDocument(); // F1-Score
    expect(within(clorofilaItem).getByText(/0.9/)).toBeInTheDocument(); // ROC AUC

    // Opcional: Podemos hacer lo mismo para las otras predicciones
    const cianoItem = screen.getByText(/Cianobacterias/i).closest('li');
    expect(within(cianoItem).getByText(/MLP/i)).toBeInTheDocument();

    // Verificar que la llamada POST se hizo con los datos correctos
    expect(axios.post).toHaveBeenCalledWith('/api/predict', {
      option: 'C1',
    });
  });
});