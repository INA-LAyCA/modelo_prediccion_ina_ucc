describe('Flujo Principal de la Aplicación', () => {

    it('debería permitir al usuario navegar, hacer una predicción y ver los resultados', () => {
      
      // --- INICIO DE LA CORRECCIÓN ---

      // Intercepción 1: La petición para obtener las opciones del dropdown.
      // Le damos datos de prueba para que la prueba sea consistente.
      cy.intercept('GET', '**/api/get-options', {
        statusCode: 200,
        body: ['C1', 'TAC1', 'DSA1']
      }).as('getOptionsRequest'); // Le damos un alias a esta petición

      // Intercepción 2: La petición para obtener la predicción.
      cy.intercept('POST', '**/api/predict', {
        statusCode: 200,
        body: [{
          codigo_perfil: 'C1',
          fecha_prediccion: '2025-08-01',
          Clorofila: { prediccion: 'Emergencia', modelo_usado: 'Cypress-Model' }
          // ... otros datos de predicción ...
        }],
      }).as('predictRequest');

      // --- FIN DE LA CORRECCIÓN ---

      // Visitar la página principal
      cy.visit('http://localhost:3000');
      
      // Navegar a la página de predicción
      cy.contains('button', 'Predecir').click();
      cy.url().should('include', '/predict');

      // ¡PASO CLAVE! Esperar a que la carga de opciones termine ANTES de continuar.
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
      cy.contains('h2', 'Datos del DataFrame Procesado');
      
      cy.visit('http://localhost:3000');
  
      cy.contains('button', 'Ver Predicciones').click();
  
      cy.url().should('include', '/predicciones');
      cy.contains('h2', 'Historial de predicciones');
  
    });
  });
  