from flask import Flask
from flask_cors import CORS
from flask_restx import Api

from app.main.controller.file_controller import File

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version='0.1',
    title = 'Oriental Chicken cuRry',
    description='Oriental Chicken cuRry api server',
    terms_url='/',
    contact='',
    license="MIT"
)

api.add_namespace(File, '/file')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
