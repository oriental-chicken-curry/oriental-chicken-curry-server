from flask import Flask
from flask_restx import Api
from app.main.controller.file_controller import File
from app.main.controller.todo import Todo

app = Flask(__name__)
api = Api(
    app,
    version='0.1',
    title = 'ganta API Server',
    description='ganta todo api server',
    terms_url='/',
    contact='ganta@naver.com',
    license="MIT"
)

api.add_namespace(Todo, '/todos')
api.add_namespace(File, '/file')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)