from flask import Flask, request
from chat import init_chat

app = Flask(__name__)
model = init_chat()

@app.route('/chat')
def chat():
    query = request.args.get('query')
    print('\n' + query)
    return model.run(query)

if __name__ == '__main__':
    app.run(debug=True)
