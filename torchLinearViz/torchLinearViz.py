import torch
from .analyse_graph import analyse_graph
from .visualization.serverSocket import start_server, stop_server  # サーバーの開始と停止関数
import threading
import json


class TorchLinearViz:
    def __init__(self, model, input_tensor):
        """
        モデルと入力テンソルを受け取り、初期化。
        """
        self.model = model
        self.input_tensor = input_tensor
        self.json_data_list = []
        self.epoch = 0
        self.json_data = None
        self.server_thread = None
#        self.hook_handles = []
        self.is_running = False

        # 初期グラフを抽出
        self.extract_and_save_graph(self.model)

    def extract_and_save_graph(self, model):
        """
        モデルのグラフ構造を抽出し、JSON形式で保存。
        """
        self.json_data = analyse_graph(model, self.input_tensor)
        with open('/path/to/output.json', 'w') as f:
            json.dump(self.json_data, f, indent=4)
            
        saveDict = {
                "epoch": self.epoch,
                "graph": self.json_data
        }
        self.json_data_list.append(saveDict)
        self.epoch += 1


    def start(self, host='0.0.0.0', port=5000, browser=True):
        """
        サーバーを開始して、Webブラウザを起動。
        """
        self.is_running = True
        self.server_thread = threading.Thread(
            target=start_server, args=(host, port, browser), daemon=True
        )
        self.server_thread.start()

        # モデルにフックを登録
#        self.register_hooks()
    '''
    def register_hooks(self):
        def hook_fn(module, inputs, outputs):
            """
            フックが呼ばれるたびにJSONデータを更新。
            """
            self.extract_and_save_graph()

        # モデルのすべてのモジュールにフックを登録
        for module in self.model.modules():
            handle = module.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)
    '''
    def update(self, model):
        """
        モデルのグラフを再解析して更新する
        """
        self.extract_and_save_graph(model)
        print("Graph updated!")

    def end(self):
        """
        サーバーを停止し、フックを解除。
        """
        self.is_running = False

        epoch_graphs_json = json.dumps(self.json_data_list)
        print('>>> ', self.json_data_list)
        html_template = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualization per epoch</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <script src="https://cdn.rawgit.com/cpettitt/dagre/v0.7.4/dist/dagre.min.js"></script>
    <script src="https://cdn.rawgit.com/cytoscape/cytoscape.js-dagre/1.5.0/cytoscape-dagre.js"></script>
    <style>
        #cy {{ width: 100%; height: 600px; border: 1px solid black; }}
        #controls {{ margin: 10px; }}
    </style>
</head>
<body>

    <h2>Visualization per epoch</h2>
    
    <div id="controls">
        <label for="epoch-slider">Epoch : <span id="epoch-label">0</span></label>
        <input type="range" id="epoch-slider" min="0" max="9" value="0" step="1">
	<button id="play-button">▶ Start Video</button>  <!-- 再生ボタンを追加 -->
    </div>

    <div id="cy"></div>

    <script>
        // Python から埋め込んだエポックデータ
        let epochData = {epoch_graphs_json};

        let cy;
        let epochSlider = document.getElementById("epoch-slider");
        let epochLabel = document.getElementById("epoch-label");
	let playButton = document.getElementById("play-button");
	let isPlaying = false;
        let playInterval = null;

        function updateGraph(epochIndex) {{
            let graph = epochData[epochIndex].graph;
            if (!cy) {{
                cy = cytoscape({{
                    container: document.getElementById("cy"),
                    elements: graph,
                    style: [
                        {{ selector: 'node', style: {{ 'label': 'data(id)', 'background-color': '#0074D9', 'shape': 'rectangle'}} }},
                        {{ selector: 'edge', style: {{ 'width': 'mapData(width, 0, 5, 1, 5)', 'line-color': '#666', 'curve-style': 'bezier' }} }}
                    ],
                    layout: {{ name: 'dagre', rankSep: 100, nodeSep: 100 }}
                    }});
                }} else {{
                cy.elements().remove();
                cy.add(graph);
                cy.layout({{ name: 'dagre', rankSep: 100, nodeSep: 100 }}).run();
                }}

            epochLabel.textContent = `${{epochIndex + 1}}/${{epochData.length - 1}}`;
            }}

        // スライダーイベント
        epochSlider.addEventListener("input", function() {{
            updateGraph(parseInt(this.value));
        }});

	    // 自動再生機能
        function playAnimation() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                isPlaying = false;
                playButton.textContent = "▶ Start Video";
            }} else {{
                let currentEpoch = parseInt(epochSlider.value);
                playButton.textContent = "■ Pause";
                isPlaying = true;

                playInterval = setInterval(() => {{
                    if (currentEpoch >= epochSlider.max) {{
                        clearInterval(playInterval);
                        isPlaying = false;
                        playButton.textContent = "▶ Start Video";
                        }} else {{
                        currentEpoch++;
                        epochSlider.value = currentEpoch;
                        updateGraph(currentEpoch);
                        }}
                }}, 1000); // 1秒ごとに次のエポックを表示
            }}
        }}

        // 再生ボタンイベント
        playButton.addEventListener("click", playAnimation);

        // 初期表示
        updateGraph(0);
    </script>

</body>
</html>
"""     # HTML ファイルとして保存
        output_html = "epoch_visualizer.html"
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_template)

#        # サーバー停止
#        if self.server_thread:
#            stop_server()  # サーバーを停止するための関数
#            self.server_thread.join()

        # フックを解除
#        for handle in self.hook_handles:
#            handle.remove()
#        self.hook_handles = []

        print("torchLinearViz has been stopped.")

