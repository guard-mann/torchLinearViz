from flask import Flask, jsonify, send_from_directory
import os
import json

app = Flask(__name__)

# JSONデータを提供するエンドポイント
@app.route('/get-graph', methods=['GET'])
def get_graph():
    # JSONファイルのパスを指定
    json_path = '/app/resource/output.csv'  # 必要に応じてフルパスを記載
    if not os.path.exists(json_path):
        return jsonify({"error": "JSON file not found"}), 404

    # JSONファイルを読み込み
    with open(json_path, 'r') as f:
        data = json.load(f)

    return jsonify(data)

# 静的ファイルを提供
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

