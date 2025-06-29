// tests/app_flow.spec.js
const { test, expect } = require('@playwright/test');

test('Flujo completo de predicción', async ({ page }) => {
  // 1. Iniciar la aplicación y el backend en modo de desarrollo por separado
  // (npm start en el frontend, python backend.py en el backend)

  // 2. Navegar a la página de inicio
  await page.goto('http://localhost:3000/'); // Asume que tu React corre en el puerto 3000

  // 3. Verificar que la página de inicio carga correctamente
  await expect(page.getByRole('heading', { name: /MODELO DE PREDICCIÓN/i })).toBeVisible();

  // 4. Hacer clic en el botón para ir a la página de predicción
  await page.getByRole('button', { name: 'Predecir' }).click();

  // 5. Verificar que se navegó a la página correcta
  await expect(page).toHaveURL(/.*predict/);
  await expect(page.getByRole('heading', { name: /Realizar Predicción/i })).toBeVisible();

  // 6. Interactuar con el selector
  // Playwright espera automáticamente a que las opciones carguen desde la API real
  await page.getByRole('combobox').selectOption('C1');

  // 7. Hacer clic en el botón de predecir
  await page.getByRole('button', { name: 'Predecir' }).click();

  // 8. Verificar que el resultado aparece en la pantalla
  // La prueba esperará a que el elemento aparezca después de la llamada a la API
  const resultsContainer = page.locator('.prediction-tab');
  await expect(resultsContainer.getByText(/Resultados para C1/i)).toBeVisible();
  await expect(resultsContainer.getByText('Alerta')).toBeVisible(); // Busca la predicción de clorofila
});