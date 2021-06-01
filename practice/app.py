from flask_restx import Api, Resource
from flask import Flask, jsonify

app = Flask(__name__)
api = Api(app)

@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {"hello" : "world"}

# request mapping
@api.route('/hi')
class HiWorld(Resource):
    def get(self):
        return jsonify({'name': 'alice',
                        'email': 'alice@outlook.com'})

@api.route('/hello2/<String:name>')
class Hello(Resource):
    def get(self, name):
        return {'message': "Welcome, %s!" % name}

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 80)