const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api', // La "palabra clave" que activa el proxy
    createProxyMiddleware({
      // La dirección de tu servidor backend
      target: 'http://localhost:5001',
      // Esto es necesario para que el proxy funcione correctamente
      changeOrigin: true,
      // ¡ESTA ES LA LÍNEA MÁGICA!
      // Le dice al proxy que reescriba la ruta antes de enviarla.
      // Elimina '/api' de la URL.
      // Así, una llamada a '/api/get-options' se convierte en '/get-options'
      // para que tu backend de Flask la entienda.
      pathRewrite: {
        '^/api': '', 
      },
    })
  );
};
