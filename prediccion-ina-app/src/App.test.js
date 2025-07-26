import React from 'react';
import { render, screen, fireEvent, waitFor, within, act  } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';
import { MemoryRouter } from 'react-router-dom';

import AppRoot, { Predict, formatDate, PredictionDetails } from './App';

// Mockear axios para controlar las respuestas de la API
jest.mock('axios');

// --- Prueba Unitaria para una función de utilidad ---
describe('formatDate utility function', () => {
  test('should format YYYY-MM-DD to DD/MM/YYYY', () => {
    expect(formatDate('2025-07-28')).toBe('28/07/2025');
  });

  test('should return "Fecha inválida" for incorrect formats', () => {

    expect(formatDate('28-07-2025')).toBe('Fecha inválida');
  });

  test('should return "-" for null or undefined input', () => {
    expect(formatDate(null)).toBe('-');
  });
});



describe('Predict Component', () => {
  test('fetches options and makes a prediction on button click', async () => {
    // 1. respuestas simuladas de la API
    const mockOptions = ['C1', 'TAC1', 'DSA1'];
    const mockPrediction = [{
      codigo_perfil: 'C1',
      fecha_prediccion: '2025-07-01T00:00:00', 
      Clorofila: { prediccion: 'Alerta', modelo_usado: 'RandomForest', f1_score_cv: 0.8, roc_auc_cv: 0.9, precision_weighted_cv: 0.7 },
      Cianobacterias: { prediccion: 'Vigilancia', modelo_usado: 'MLP', f1_score_cv: 0.85, roc_auc_cv: 0.92, precision_weighted_cv: 0.95 },
      Dominancia: { prediccion: 'No Dominante', modelo_usado: 'LogisticRegression', f1_score_cv: 0.9, roc_auc_cv: 0.95, precision_weighted_cv: 0.95 }
    }];

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
    await waitFor(() => {
      expect(screen.getByText(/Resultados para C1/i)).toBeInTheDocument();
      expect(screen.getByText('Alerta')).toBeInTheDocument();
    });

  
    const clorofilaItem = screen.getByText(/^Clorofila:$/i).closest('li');
    expect(within(clorofilaItem).getByText(/RandomForest/i)).toBeInTheDocument();
    expect(within(clorofilaItem).getByText(/0.7/)).toBeInTheDocument();
    const cianoItem = screen.getByText(/^Cianobacterias:$/i).closest('li');
    expect(within(cianoItem).getByText(/MLP/i)).toBeInTheDocument();
  
    expect(axios.post).toHaveBeenCalledWith('/api/predict', {
      option: 'C1',
    });
  });
});

describe('App Component Status Polling', () => {
  
  beforeEach(() => {
    jest.useFakeTimers();
    axios.get.mockClear();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('shows and hides the retraining overlay based on API status', async () => {
    // 1. Estado inicial: El backend está inactivo ('idle')
    axios.get.mockResolvedValue({ data: { status: 'idle' } });

    render(<AppRoot />);

    expect(screen.queryByText(/Espere, los modelos están siendo actualizados/i)).toBeNull();
    // La llamada se hace una vez al montar el componente.
    expect(axios.get).toHaveBeenCalledTimes(1);
    expect(axios.get).toHaveBeenCalledWith('/api/status');
    
    // 2. Cambio de estado: El backend empieza a reentrenar ('retraining')
    axios.get.mockResolvedValue({ data: { status: 'retraining' } });

    act(() => {
      jest.advanceTimersByTime(3000);
    });

    // Esperamos a que aparezca el overlay.
    await waitFor(() => {
      expect(screen.getByText(/Espere, los modelos están siendo actualizados/i)).toBeInTheDocument();
    });
    
    expect(axios.get).toHaveBeenCalledTimes(2);

    // 3. Vuelta al estado inicial: El backend termina ('idle')
    axios.get.mockResolvedValue({ data: { status: 'idle' } });


    act(() => {
      jest.advanceTimersByTime(3000);
    });

    // Esperamos a que el overlay desaparezca.
    await waitFor(() => {
      expect(screen.queryByText(/Espere, los modelos están siendo actualizados/i)).toBeNull();
    });
   
    expect(axios.get).toHaveBeenCalledTimes(3);
  });
});
