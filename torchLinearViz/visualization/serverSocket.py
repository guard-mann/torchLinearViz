from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO
import json
import time
import os


app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def serve_html():
    return send_from_directory('static', 'websocket.html')


# クライアントが接続した際に通知
@socketio.on('connect')
def handle_connect():
    print('Client connected')


# JSONデータを定期的に送信する処理
def send_json_data():
    while True:
        # サンプルデータを生成（ここを任意の処理に置き換え可能）

        # JSONファイルのパスを指定
        json_path = '/app/resource/output.csv'  # 必要に応じてフルパスを記載
        if not os.path.exists(json_path):
            return jsonify({"error": "JSON file not found"}), 404
        # JSONファイルを読み込み
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # WebSocketでデータを送信
        socketio.emit('update_graph', data)
        time.sleep(5)  # 5秒ごとに送信

# サーバーの起動時にバックグラウンドでデータ送信を開始
@socketio.on('start')
def handle_start():
    socketio.start_background_task(send_json_data)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
