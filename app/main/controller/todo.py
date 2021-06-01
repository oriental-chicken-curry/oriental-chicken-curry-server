from flask_restx import Resource, Api, Namespace, fields

count = 0
Todo = Namespace(
    name='Todo',
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
    def get(self):
        return {"hello": "get world"}

    def post(self):
        global count

        idx = count
        count += 1
        # todos[idx] = request.json.get('data')

        return {
            'todo_id': idx,
            # 'data': todos[idx]
        }
