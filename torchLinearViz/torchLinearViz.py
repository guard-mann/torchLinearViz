from .analyse_graph import analyse_graph
import json

class TorchLinearViz:
    def __init__(self, model):
        """
        モデルと入力テンソルを受け取り、初期化。
        """
        self.model = model
        self.json_data_list = []
        self.json_data_diff_list = []
        self.epoch = 0
        self.json_data = None
        self.json_data_diff = None
        self.server_thread = None
        self.is_running = False

#        初期グラフを抽出
#        self.extract_and_save_graph(self.model)

    def extract_and_save_graph(self, model, input):
        """
        モデルのグラフ構造を抽出し、JSON形式で保存。
        """
        self.json_data, self.json_data_diff = analyse_graph(model, input, self.json_data_list)
           
        saveDict = {
                "epoch": self.epoch,
                "graph": self.json_data
        }
        saveDictDiff = {
                "epoch": self.epoch,
                "graph": self.json_data_diff
        }
        self.json_data_list.append(saveDict) # !!!
        self.json_data_diff_list.append(saveDictDiff)
        self.epoch += 1


    def update(self, model, input):
        """
        モデルのグラフを再解析して更新する
        """
        self.extract_and_save_graph(model, input)

    def end(self):
        self.is_running = False
        epoch_graphs_json = json.dumps(self.json_data_list)
        epoch_graphs_json_diff = json.dumps(self.json_data_diff_list)
        html_template = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summary | TorchLinearViz</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <script src="https://cdn.rawgit.com/cpettitt/dagre/v0.7.4/dist/dagre.min.js"></script>
    <script src="https://cdn.rawgit.com/cytoscape/cytoscape.js-dagre/1.5.0/cytoscape-dagre.js"></script>
    <script src="https://unpkg.com/webcola/WebCola/cola.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-cola@2.4.0/cytoscape-cola.min.js"></script>
    <style>
        #cy {{ width: 100%; height: 600px; border: 1px solid black; }}
        #controls {{ margin: 10px; }}
        .control-button {{
            width: 120px;  /* ボタンの横幅 */
            height: 40px;  /* ボタンの高さ */
            text-align: center;
            font-size: 16px;
        }}
        .data-switch {{
            width: 120px;  /* ボタンの横幅 */
            height: 40px;  /* ボタンの高さ */
            text-align: center;
            font-size: 16px;           
        }}
    </style>
</head>
<body>

    <h2>torchLinearViz | Animation</h2>
    
    <div id="controls">
        <button id="toggle-data", class="data-switch">Switch Data</button>
        <button id="play-button", class="control-button">▶ Start Video</button>
    </div>

    <div>
        <label for="epoch-slider">Epoch : <span id="epoch-label"></span></label>
        <input type="range" id="epoch-slider" min="0" value="0" step="1">
        <label for="speed-slider">Speed: <span id="speed-label">x1</span></label>
        <input type="range" id="speed-slider" min="1" max="10" value="1" step="1">
    </div>

    <div style="display: flex; align-items: center;">
	    <span>Weight(MIN)  </span>
	    <input type="number" id="min-value-input" value="0" step="0.01" style="width: 60px; margin-right: 10px;"> 
	    <span>Weight(MAX) </span>
	    <input type="number" id="max-value-input" value="1" step="0.01" style="width: 60px; margin: 0 10px;"> 
	    <button id="apply-width-btn">Apply</button>
    </div>

    <div>
        <button id="toggle-theme", class="theme-switch">ColorScheme</button>
    </div>

    <div id="cy"></div>

    <script> // Python から埋め込んだエポックデータ
        let epochData = {epoch_graphs_json};
        let epochData1 = {epoch_graphs_json};  // 元のデータ
        let epochData2 = {epoch_graphs_json_diff};  // 切り替え用データ
        let isUsingEpochData1 = true; // 現在のデータがどちらか判別するフラグ
        let isDarkMode = true;
        document.getElementById("toggle-data").addEventListener("click", () => {{
            isUsingEpochData1 = !isUsingEpochData1; // フラグを反転
            epochData = isUsingEpochData1 ? epochData1 : epochData2; // データを切り替え
            updateGraph(parseInt(epochSlider.value)); // 現在のエポックを再描画
            document.getElementById("toggle-data").textContent = isUsingEpochData1 ? "Weight value" : "Weight Diff";
            let button = document.getElementById("toggle-data")
            if (isUsingEpochData1) {{
                button.textContent = "Value";
                button.style.backgroundColor = "#15c222";
                button.style.color = "white";
            }} else {{
                button.textContent = "Diff (abs)";
                button.style.backgroundColor = "#FF4136";
                button.style.color = "white";
            }}

        }});


        document.getElementById("toggle-theme").addEventListener("click", () => {{
            isDarkMode = !isDarkMode; // フラグを反転

            // 背景色を変更
            document.getElementById("cy").style.backgroundColor = isDarkMode ? "#000000" : "#FFFFFF";  // 黒 or 白

            // Cytoscapeのエッジの色を変更
            cy.style()
                .selector('edge')
                .style({{
                    'line-color': isDarkMode ? '#FFFFFF' : '#666', // 白 or グレー
                    'target-arrow-color': isDarkMode ? '#FFFFFF' : '#666' // 矢印の色も変更
                }})
                .selector('node')
                .style({{
                    'color': isDarkMode ? 'white' : 'black'
                }})
                .update();

            // ボタンのラベルと色を変更
            let button = document.getElementById("toggle-theme");
            if (isDarkMode) {{
                button.textContent = "ColorScheme (Dark)";
                button.style.backgroundColor = "#444";  // ダークモードのボタン色
                button.style.color = "white";
            }} else {{
                button.textContent = "ColorScheme (Light)";
                button.style.backgroundColor = "#DDD";  // ライトモードのボタン色
                button.style.color = "black";
            }}
        }});





        let cy;
        let epochSlider = document.getElementById("epoch-slider");
        epochSlider.max = epochData.length - 1; // 最大値をデータの長さに設定
        let epochLabel = document.getElementById("epoch-label");
        epochLabel.textContent = `0/${{epochData.length - 1}}`
        let playButton = document.getElementById("play-button");
        let isPlaying = false;
        let playInterval = null;

        let speedSlider = document.getElementById("speed-slider");
        let speedLabel = document.getElementById("speed-label");
        speedSlider.min = 1;  // 1倍速
        speedSlider.max = 10; // 最大10倍速
        speedSlider.value = 1; // 初期値（通常の1倍速）
        speedLabel.textContent = `x${{speedSlider.value}}`;
        let playbackSpeed = parseInt(speedSlider.value); // 初期速度（1エポック/秒）

        let minValue = 0;
        let maxValue = 1;

        const minInput = document.getElementById("min-value-input");
        const maxInput = document.getElementById("max-value-input");
        const applyButton = document.getElementById("apply-width-btn");
        let epochNodePositions = {{}}

        // "Apply" ボタンを押したときに反映
        applyButton.addEventListener("click", function() {{
            // 入力値を取得し、数値に変換
            minValue = parseFloat(minInput.value);
            maxValue = parseFloat(maxInput.value);

            // 入力値が不正でないかチェック
            if (isNaN(minValue) || isNaN(maxValue) || minValue >= maxValue) {{
            alert("Invalid values! Ensure that minValue < maxValue and both are numbers.");
            return;
            }}

            // Cytoscape のエッジ幅を更新
            cy.style()
              .selector("edge")
              .style({{ "width": `mapData(width, ${{minValue}}, ${{maxValue}}, 1, 5)` }})
              .update();
        }})


        function updateGraph(epochIndex) {{
            let graph = epochData[epochIndex].graph;

            if (cy) {{
                zoomLevel = cy.zoom();
                panPosition = cy.pan();
            }}

            if (!cy) {{
                cy = cytoscape({{
                    container: document.getElementById("cy"),
                    elements: graph,
                    style: [
                        {{ selector: 'node', style: {{ 'label': 'data(id)', 'background-color': '#0074D9', 'shape': 'rectangle', 'width': 100, 'height': 100}} }},

                        {{ selector: 'node[type="UNIT"]',
                           style: {{
                            'background-color': '#6b6a6a',
                            "label": 'data(id)',
                            'shape': 'rectangle',
                            'font-size': 18,
                            'width': 25,
                            'height': 25,
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'text-wrap': 'wrap'
                          }}
                        }},


                        {{ selector: 'edge', style: {{ 'width': 'mapData(width, 0, 1, 1, 5)', 'line-color': '#666', 'curve-style': 'bezier' }} }}
                    ],
                    layout: {{ name: 'dagre', rankSep: 100, nodeSep: 100 }}
                    }});
                }} else {{
                cy.elements().remove();
                cy.add(graph);
                cy.layout({{ name: 'dagre', rankSep: 100, nodeSep: 100 }}).run();
                }}

            if (epochNodePositions[epochIndex]) {{
                cy.nodes().forEach(node => {{
                    if (epochNodePositions[epochIndex][node.id()]) {{
                        node.position(epochNodePositions[epochIndex][node.id()]);
                    }}
                }});
            }}

            // 📌 変更後にズーム・位置を適用
            cy.zoom(zoomLevel);
            cy.pan(panPosition);

            epochLabel.textContent = `${{epochIndex}}/${{epochData.length - 1}}`;
            }}
        cy?.on('zoom pan', function () {{
            zoomLevel = cy.zoom();
            panPosition = cy.pan();
        }});

        // スライダーイベント
        epochSlider.addEventListener("input", function() {{
            updateGraph(parseInt(this.value));
        }});

        speedSlider.addEventListener("input", function() {{
            playbackSpeed = parseInt(this.value);
            speedLabel.textContent = `x${{playbackSpeed}}`;
            if (isPlaying) {{
                clearInterval(playInterval); // **既存の再生を止めて**
                startPlayback(); // **新しい速度で即座に再開**
            }}
	    }});

        function startPlayback() {{
            let currentEpoch = parseInt(epochSlider.value);
            playInterval = setInterval(() => {{
            if (currentEpoch >= epochSlider.max) {{
                clearInterval(playInterval);
                isPlaying = false;
                playButton.textContent = "▶ Start Video";
            }} else {{
                currentEpoch++;
                if (currentEpoch > epochSlider.max) {{
                currentEpoch = epochSlider.max;
                }}
                epochSlider.value = currentEpoch;
                updateGraph(currentEpoch);
            }}
            }}, 1000 / playbackSpeed);
        }}



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

                epochNodePositions[currentEpoch] = {{}};
                for (let epochIndex = 0; epochIndex < epochData.length; epochIndex++){{
                    if (!epochNodePositions[epochIndex]) {{
                    epochNodePositions[epochIndex] = {{}};
                    }}
                    cy.nodes().forEach(node => {{
                    epochNodePositions[epochIndex][node.id()] = node.position();
                    }}); 
                }}

                playInterval = setInterval(() => {{
                    if (currentEpoch >= epochData.length) {{
                        clearInterval(playInterval);
                        isPlaying = false;
                        playButton.textContent = "▶ Start Video";
                        }} else {{
                        currentEpoch++;
                        epochSlider.value = currentEpoch;
                        updateGraph(currentEpoch);
                        }}
                }}, 1000 / playbackSpeed ); // 1秒ごとに次のエポックを表示
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

        print("torchLinearViz has been finished.")

