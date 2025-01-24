// サーバーのエンドポイントにアクセスしてJSONデータを取得
fetch('/get-graph')
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to fetch JSON data');
        }
        return response.json();
    })
    .then(data => {
        // JSONデータをHTMLに描画
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    })
    .catch(error => {
        console.error("Error fetching JSON data:", error);
        document.getElementById('output').textContent = "Error loading data.";
    });

