from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO
import socket
import json
import time
import os
import webbrowser
from threading import Thread


app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def serve_html():
    return send_from_directory('static', 'websocket.html')


# クライアントが接続した際に通知
@socketio.on('connect')
def handle_connect():
    print('Client connected')


def find_free_port(default_port=5000):
    """
    空いているポートを見つける。
    :param default_port: デフォルトで試みるポート番号
    :return: 使用可能なポート番号
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if s.connect_ex(('localhost', default_port)) != 0:
            return default_port  # デフォルトポートが空いている
        # デフォルトポートが使われている場合、別のポートを探す
        for port in range(default_port + 1, default_port + 100):
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise RuntimeError("No free ports available")

# JSONデータを定期的に送信する処理
def send_json_data():
    while True:
        # サンプルデータを生成（ここを任意の処理に置き換え可能）

        # JSONファイルのパスを指定
        json_path = '/path/to/your/json'  # 必要に応じてフルパスを記載
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

def start_server(host='0.0.0.0', port=5000, browser=False):
    free_port = find_free_port(port)
    if browser:
        url = f"http://localhost:{free_port}"
        Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        
    socketio.run(app, host=host, port=free_port)
