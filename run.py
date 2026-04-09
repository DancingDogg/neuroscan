import os
from dotenv import load_dotenv
from app import create_app

load_dotenv()
secret_key = os.environ.get('SECRET_KEY')

if not secret_key:
    raise RuntimeError("SECRET_KEY not found in environment variables. Please set it in a .env file.")

if __name__ == '__main__':
    app = create_app(secret_key)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true', host='0.0.0.0', port=5000)