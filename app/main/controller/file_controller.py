import cv2
import numpy

from flask_restx import Resource, Namespace
from werkzeug.datastructures import FileStorage

from app.main.service.inference import *

File = Namespace(
    name='File',
    description="upload & download file"
)

upload_parser = File.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


@File.route('')
@File.expect(upload_parser)
class uploadFile(Resource):
    """ 파일 컨트롤러 정의
    """
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        uploaded_file = uploaded_file.read()
        npimg = numpy.fromstring(uploaded_file, numpy.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        res = inference(img)
        return res
