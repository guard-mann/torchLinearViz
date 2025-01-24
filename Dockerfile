# ベースイメージ: PyTorchの公式イメージを使用
FROM pytorch/pytorch

# 作業ディレクトリを作成
WORKDIR /app

# 必要なファイルをコピー
COPY requirements.txt ./

# Pythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトのコードをコピー
COPY . .

# ポートを公開（WebSocketサーバー用）
EXPOSE 8765

# デフォルトの起動コマンド
# CMD ["python", "server.py"]
