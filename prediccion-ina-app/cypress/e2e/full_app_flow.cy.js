describe('Flujo Principal de la Aplicación', () => {

    it('debería permitir al usuario navegar, hacer una predicción y ver los resultados', () => {
      
      cy.intercept('GET', '**/api/get-options', {
        statusCode: 200,
        body: ['C1', 'TAC1', 'TAC4', 'DCQ1','DSA1']
      }).as('getOptionsRequest'); 

      cy.intercept('POST', '**/api/predict', {
        statusCode: 200,
        body: [{
          codigo_perfil: 'C1',
          fecha_prediccion: '2025-08-01',
          Clorofila: { prediccion: 'Emergencia', modelo_usado: 'Cypress-Model' }
          
        }],
      }).as('predictRequest');

  

      // Visitar la página principal
      cy.visit('http://localhost:3000');

      const checkBackendStatus = (retries = 30) => { // 30 reintentos * 10s = 5 minutos de espera máxima
        if (retries < 0) {
          throw new Error('El backend no estuvo listo a tiempo.');
        }
        cy.request({
          url: '/api/status', // URL completa del backend
          failOnStatusCode: false // No fallar si la API devuelve un error temporalmente
        }).then((response) => {
          // Si el backend responde que está 'idle' (inactivo), continuamos.
          if (response.status === 200 && response.body.status === 'idle') {
            cy.log('Backend listo. Continuando con la prueba.');
          } else {
            // Si no, esperamos 10 segundos y volvemos a intentar.
            cy.log(`Backend ocupado o no disponible... Reintentos restantes: ${retries}`);
            cy.wait(10000); 
            checkBackendStatus(retries - 1);
          }
        });
      }
      
      cy.log('Esperando a que el backend esté listo...');
      checkBackendStatus();  

      // Navegar a la página de predicción
      cy.contains('button', 'Predecir').click();
      cy.url().should('include', '/predict');

      // Esperar a que la carga de opciones termine ANTES de continuar.
      cy.wait('@getOptionsRequest');

      // Ahora que las opciones están cargadas, podemos interactuar con el formulario.
      cy.get('select').select('C1');
      cy.contains('button', 'Predecir').click();

      // Esperar a que la petición de predicción se complete.
      cy.wait('@predictRequest');

      // Verificar que los resultados se muestran correctamente.
      cy.contains('Resultados para C1').should('be.visible');
      cy.contains('.prediction-label', 'Emergencia').should('be.visible');
      cy.visit('http://localhost:3000');
  
      cy.contains('button', 'Ver Datos').click();
  
      cy.url().should('include', '/datos');
      cy.contains('h1', 'Tabla de Datos Procesado');
      
      cy.visit('http://localhost:3000');
  
      cy.contains('button', 'Ver Predicciones').click();
  
      cy.url().should('include', '/predicciones');
      cy.contains('h1', 'Historial de predicciones');
  
    });
  });
  