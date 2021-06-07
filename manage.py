from flask import Flask
from flask_restx import Api
from app.main.controller.file_controller import File
from app.main.controller.todo import Todo
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version='0.1',
    title = 'ganta API Server',
    description='ganta todo api server',
    terms_url='/',
    contact='ganta@naver.com',
    license="MIT"
)

api.add_namespace(Todo, '/todos')# 튜토리얼
api.add_namespace(File, '/file')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
