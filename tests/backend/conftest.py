# tests/backend/conftest.py
import pytest
from backend import app as flask_app # Importamos la instancia de Flask

@pytest.fixture
def app():
    """Crea y configura una nueva instancia de la app para cada prueba."""
    # Configuración adicional de prueba si es necesaria
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app

@pytest.fixture
def client(app):
    """Un cliente de prueba para la app."""
    return app.test_client()