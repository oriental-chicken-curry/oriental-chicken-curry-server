import numpy
from flask import Flask, jsonify, request
from flask_restx import Resource, Api, Namespace
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import cv2

from practice.todo import Todo

# http://172.30.1.56

app = Flask(__name__)
api = Api(
    app,
    version='0.1',
    title = 'heesup API Server',
    description='heesup todo api server',
    terms_url='/',
    contact='ganta@naver.com',
    license="MIT"
)

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


@api.route('/upload/')
@api.expect(upload_parser)
class Upload(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        uploaded_file = uploaded_file.read()
        npimg = numpy.fromstring(uploaded_file, numpy.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        print(type(img))
        print(img.shape)
        # url = do_something_with_file(uploaded_file)
        return "success"

# # request mapping
# @api.route('/hi')
# class HiWorld(Resource):
#     def get(self):
#         return jsonify({'name': 'alice',
#                         'email': 'alice@outlook.com'})

# @api.route('/hello/<string:name>')  # url pattern으로 name 설정
# class Hello(Resource):
#     def get(self, name):  # 멤버 함수의 파라미터로 name 설정
#         return {"message": "Welcome, %s!" % name}

# @api.route('/hello')
# class HelloWorld(Resource):
#     def get(self):
#         return {"hello" : "get world"}
#
#     def post(self):
#         return{"hello" : "post world"}
#
#     def put(self):
#         return {"hello": "put world"}
#
#     def delete(self):
#         return {"delete": "success"}
#
# namespace = Namespace('hello')
#
#
#
# @api.route('/hello/<string:name>')
# class Hello(Resource):
#     def get(self, name):
#         return {"message" : "Welcome, %s!" % name}

# -------------------------------------------
# namespace = Namespace('hi')  # 첫 번째
#
# @namespace.route('/') # namespace route 설정
# class HelloWorld(Resource):
#     def get(self):
#         return {"hello" : "world!"}, 201, {"hi":"hello"}
#
# api.add_namespace(namespace, '/hello')
#
#
# @api.route('/hello')  # default route설정
# class HelloWorld(Resource):
#     def get(self):
#         return {"hello" : "world!"}, 201, {"hi":"hello"}
# -------------------------------------------

# 파일 업로드 처리
@app.route('/fileUpload',methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return "success file upload"

api.add_namespace(Todo, '/todos')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)