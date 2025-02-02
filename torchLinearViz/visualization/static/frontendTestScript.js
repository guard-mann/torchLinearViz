// WebSocket サーバーに接続
const socket = io.connect('http://localhost:5000');

const viewportWidth = window.innerWidth;
const viewportHeight = window.innerHeight;


// Cytoscape.js でグラフを描画
let cy = cytoscape({
  container: document.getElementById('cy'), // 描画する HTML 要素
  elements: [], // JSON データをそのまま渡す
  style: [ // スタイル設定
    {
      selector: 'node',
      style: {
	'label': function (ele) {
          return ele.data('id') + '\n(' + ele.data('type') + ')'; // IDとタイプを結合
        },
	'shape': 'rectangle',
	'font-size': 18,
        'width': 100,
        'height': 100,
        'text-valign': 'center',
        'text-halign': 'center',
	'text-wrap': 'wrap'
      }
    },
    {
      selector: 'node[type="Linear"]',
      style: {
        'background-color': '#1264bb',// https://tools.tolog.info/tools/color-code-converter より
	'label': function (ele) {
          return ele.data('id') + '\n(' + ele.data('type') + ')'; // IDとタイプを結合
        },
	'shape': 'rectangle',
	'font-size': 18,
        'width': 100,
        'height': 100,
        'text-valign': 'center',
        'text-halign': 'center',
	'text-wrap': 'wrap'
      }
    },
    {
      selector: 'node[type="MaxPool2d"], node[type="AdaptiveAvgPool2d"]',
      style: {
        'background-color': '#b82b17',
	'label': function (ele) {
          return ele.data('id') + '\n(' + ele.data('type') + ')'; // IDとタイプを結合
        },
	'shape': 'rectangle',
	'font-size': 18,
        'width': 100,
        'height': 100,
        'text-valign': 'center',
        'text-halign': 'center',
	'text-wrap': 'wrap'
      }    
    },
    {
      selector: 'node[type="Conv2d"], node[type="Conv3d"]',
      style: {
        'background-color': '#17b81b',
	'label': function (ele) {
          return ele.data('id') + '\n(' + ele.data('type') + ')'; // IDとタイプを結合
        },
	'shape': 'rectangle',
	'font-size': 18,
        'width': 100,
        'height': 100,
        'text-valign': 'center',
        'text-halign': 'center',
	'text-wrap': 'wrap'
      }    
    },
    {
      selector: 'node[type="UNIT"]',
      style: {
	'label': function (ele) {
          return ele.data('id') + '\n(' + ele.data('type') + ')'; // IDとタイプを結合
        },
	'font-size': 18,
        'width': 25,
        'height': 25,
        'text-valign': 'center',
        'text-halign': 'center',
	'text-wrap': 'wrap'
      }    
    },
    {
      selector: 'edge',
      style: {
        'width': 'mapData(width, 0, 1, 1, 5)',
        'line-color': '#090808',
        'target-arrow-color': '#090808',
        'target-arrow-shape': 'triangle',
	'curve-style': 'bezier',
	'opacity': 0.5
      }
    }
  ],
  layout: { // グラフのレイアウト設定
    name: 'dagre',
    rankSep: 500,
    nodeSep: 500,
  },
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
  console.log("Recieved updated data :", data);
  console.log("Received data:", data);
  if (!initialized) {
    cy.json({ elements: data });

    // 初回だけレイアウトを実行
    cy.layout({
      name: 'dagre',
      rankSep: 500,
      nodeSep: 500,
    }).run();
    initialized = true;
  } else {
    // 初回以降はデータを更新するだけでレイアウトを再実行しない
    console.log("Adding New edges: ", data.edges);
    cy.json({elements: data});
    cy.layout({
      name: 'dagre',
      rankSep: 500,
      nodeSep: 500,
    }).run();
  }
});

socket.emit('start')
