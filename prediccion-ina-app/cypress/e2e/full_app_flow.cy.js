describe('Flujo Principal de la Aplicación', () => {

    it('debería permitir al usuario navegar, hacer una predicción y ver los resultados', () => {
      
      // CAMBIO CLAVE AQUÍ: Usamos un patrón glob más robusto.
      cy.intercept('POST', '**/api/predict', {
        statusCode: 200,
        body: [{
          codigo_perfil: 'C1',
          fecha_prediccion: '2025-08-01',
          Clorofila: { prediccion: 'Emergencia', modelo_usado: 'Cypress-Model', f1_score_cv: 0.99, roc_auc_cv: 0.99 },
          Cianobacterias: { prediccion: 'Vigilancia', modelo_usado: 'Cypress-Model', f1_score_cv: 0.9, roc_auc_cv: 0.9 },
          Dominancia: { prediccion: 'No Dominante', modelo_usado: 'Cypress-Model', f1_score_cv: 0.95, roc_auc_cv: 0.96 }
        }],
      }).as('predictRequest');
  
      cy.visit('http://localhost:3000');
  
      cy.contains('button', 'Predecir').click();
  
      cy.url().should('include', '/predict');
      cy.contains('h2', 'Realizar Predicción').should('be.visible');
  
      cy.get('select').select('C1');
      cy.contains('button', 'Predecir').click();
  
      // Ahora la espera debería funcionar correctamente
      cy.wait('@predictRequest');
  
      cy.contains('Resultados para C1').should('be.visible');
      cy.contains('.prediction-label', 'Emergencia').should('be.visible');
      cy.contains('.metrics', 'Cypress-Model').should('be.visible');
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
  