"""
WSGI entry point
"""
import os
from backend.main import app

if __name__ == '__main__':
    if not os.path.exists('backend/uploads'):
        os.mkdir('backend/uploads')
    app.run()
