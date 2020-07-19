from flask import Flask
from flask import request, jsonify
import json
app = Flask(__name__)


@app.route('/')
def hello_world():
    data = request.get_json()
    print(data)
    code = data.get("code")
    start_date = data.get("startdate")
    return json.dumps('Hello World!', ensure_ascii=False, indent=4)


if __name__ == '__main__':
    app.run()
