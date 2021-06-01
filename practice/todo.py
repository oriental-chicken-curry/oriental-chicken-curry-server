from flask import request
from flask_restx import Resource, Api, Namespace, fields

todos = {}
count = 1

Todo = Namespace(
    name = 'Todo',
    description="test"
)

todo_fields = Todo.model('Todo', {  # Model 객체 생성
    'data': fields.String(description='a Todo', required=True, example="what to do")
})

todo_fields_with_id = Todo.inherit('Todo With ID', todo_fields, {
    'todo_id': fields.Integer(description='a Todo ID')
})

@Todo.route('')
class TodoPost(Resource):
    def post(self):
        global count
        global todos

        idx = count
        count += 1
        # todos[idx] = request.json.get('data')

        return {
            'todo_id': idx,
            # 'data': todos[idx]
        }


@Todo.route('/<int:todo_id>')
@Todo.doc(params = {'todo_id' : 'An ID'}) # 파라미터 설명
class TodoSimple(Resource):
    @Todo.response(200, 'Success', todo_fields_with_id)
    @Todo.response(500, 'Failed')
    def get(self, todo_id):
        """get test"""
        return {
            'todo_id': todo_id,
            # 'data': todos[todo_id]
        }

    @Todo.response(202, 'Success', todo_fields_with_id)
    @Todo.response(500, 'Failed')
    def put(self, todo_id):
        """put test"""
        # todos[todo_id] = request.json.get('data')
        return {
            'todo_id': todo_id,
            # 'data': todos[todo_id]
        }

    @Todo.doc(responses = {202: 'Success'})
    @Todo.doc(responses={500: 'Fail'})
    def delete(self, todo_id):
        """delete test"""
        # del todos[todo_id]
        return {
            "delete": "success"
        } , 202 #코드 반환