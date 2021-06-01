import numpy
import cv2
from flask_restx import Resource, Api, Namespace, fields
from werkzeug.datastructures import FileStorage

File = Namespace(
    name = 'File',
    description="upload & download file"
)

todo_fields = File.model('Todo', {  # Model 객체 생성
    'data': fields.String(description='a Todo', required=True, example="what to do")
})

todo_fields_with_id = File.inherit('Todo With ID', todo_fields, {
    'todo_id': fields.Integer(description='a Todo ID')
})


upload_parser = File.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

@File.route('')
@File.expect(upload_parser)
class uploadFile(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        uploaded_file = uploaded_file.read()
        npimg = numpy.fromstring(uploaded_file, numpy.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        print("파일 타입 : ", type(img))
        print("이미지 shape : ",img.shape)
        # Todo 모델 서빙 로직 추가
        return "success"