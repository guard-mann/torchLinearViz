// WebSocket サーバーに接続
const socket = io.connect('http://localhost:5000');

// Cytoscape.js でグラフを描画
let cy = cytoscape({
  container: document.getElementById('cy'), // 描画する HTML 要素
  elements: [], // JSON データをそのまま渡す
  style: [ // スタイル設定
    {
      selector: 'node',
      style: {
        'background-color': '#0074D9',
        'label': 'data(id)', // ノードの ID をラベルに表示
        'width': 10,
        'height': 10
      }
    },
    {
      selector: 'edge',
      style: {
        'width': 'mapData(width, 1, 10, 1, 5)',
        'line-color': '#AAAAAA',
        'target-arrow-color': '#AAAAAA',
        'target-arrow-shape': 'triangle'
      }
    }
  ],
  layout: { // グラフのレイアウト設定
    name: 'cose', // 力学モデルに基づくレイアウト
    idealEdgeLength: 100,
    nodeRepulsion: 10000,
    animate: true,
    randomized: true,
    seed: 42
  }
});

socket.on('connect', () => {
    console.log('WebSocket connected');
});

socket.on('disconnect', () => {
    console.log('WebSocket disconnected');
});

let initialized = false;

// WebSocket からデータを受信
socket.on('update_graph', (data) => {

  console.log("Received data:", data);
  if (!initialized) {
    // 初回だけレイアウトを実行
    cy.json({ elements: data });
    cy.layout({
      name: 'cose',
      idealEdgeLength: 50,
      nodeRepulsion: 10000,
      animate: true
    }).run();
    initialized = true;
  } else {
    // 初回以降はデータを更新するだけでレイアウトを再実行しない
    cy.add(data.nodes);
    cy.add(data.edges);
  }
  
  cy.edges().forEach(edge => {
    const edgeData = data.edges.find(e => e.data.id === edge.id());
    if (edgeData) {
      edge.style('width', edgeData.data.width); // サーバーからの新しい width を適用
    }
  });

});

socket.emit('start')
