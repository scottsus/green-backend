from flask import Flask, request, jsonify
from flask_cors import CORS
from chat import init_chat, analyze_statement

app = Flask(__name__)
CORS(app)
model, parser = init_chat()

@app.route('/chat', methods=['GET'])
def chat():
    query = request.args.get('query')
    if not query:
      return jsonify({"error": "Query parameter missing."}), 400

    try:
        analyzed = analyze_statement(model, parser, query)
        return jsonify([analyzed.dict()])
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
