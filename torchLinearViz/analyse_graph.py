import torch
import torch.nn as nn

def convert_to_json_format(nodes, edges):
    return {"nodes": nodes, "edges": edges}

def analyse_graph(model, input_tensor, existing_graphs):
    edges = []
    edgeList = []
    edgeDiffList = []
    nodeList = []
    adjustNode = []
    layer_names = {}
    last_valid_layer = None  # 直前の有効なレイヤーを追跡
    tensor_to_layer = {}  # テンソル ID からレイヤー名へのマッピング
    count_edge = 0

    # フォワードフックを使ってレイヤー間の接続を記録
    def hook_fn(module, input, output):
        nonlocal last_valid_layer
        layer_name = layer_names[module]  # レイヤー名を取得

        if isinstance(output, torch.Tensor):

            output_id = id(output)  # 出力テンソルの ID を取得

            # 直前の有効なレイヤーが存在すればエッジを追加
            if last_valid_layer is not None:
                edges.append((last_valid_layer, layer_name))  # (前のレイヤー → 現在のレイヤー)
            
            # テンソル ID をレイヤー名と紐付ける
            tensor_to_layer[output_id] = layer_name
            last_valid_layer = layer_name  # 最後に処理したレイヤーを更新

    # 各レイヤーにフックを登録
    skipRateIn = 1
    skipRateOut = 1
    MAXNODE = 25
    for name, layer in model.named_modules():
    #    if isinstance(layer, (nn.Linear, nn.ReLU)):  # 追加したいレイヤー
        layer_names[layer] = name  # レイヤー名を保存
        layer.register_forward_hook(hook_fn)  # フックを登録
        if isinstance(layer, nn.Linear):  # 全結合層のみ
            skipRateIn = 1
            adjustNode.append(name)
            linearOutName = name + "_out"
            if len(name) > 0:
                nodeList.append({"data": {"id": name, "type": layer.__class__.__name__}})
            nodeList.append({"data": {"id": linearOutName, "type": layer.__class__.__name__}})
            if layer.in_features > MAXNODE:
                skipRateIn = round( layer.in_features / MAXNODE)
            if layer.out_features > MAXNODE:
                skipRateOut = round( layer.out_features / MAXNODE)

            for input_unit in range(layer.in_features):
                count_edge += 1

                if input_unit % skipRateIn == 0:
                    unitInName = "UNIT_" + name + "_in_" + str(input_unit)

                    nodeList.append({"data": {"id": unitInName, "type": "UNIT"}})
                    edgeList.append({"data": {"id": count_edge, "source": name, "target": unitInName, "width": 1}})
            for output_unit in range(layer.out_features):
                if output_unit % skipRateOut == 0:
                    count_edge += 1
                    unitOutName = "UNIT_" + name + "_out_" + str(output_unit)
                    nodeList.append({"data": {"id": unitOutName, "type": "UNIT"}})
                    edgeList.append({"data": {"id": count_edge, "source": unitOutName, "target": linearOutName, "width": 1}})
            for input_unit in range(layer.in_features):
                unitInName = "UNIT_" + name + "_in_" + str(input_unit)
                if input_unit % skipRateIn == 0:
                    for output_unit in range(layer.out_features):
                        if output_unit % skipRateOut == 0:
                            count_edge += 1
                            unitOutName = "UNIT_" + name + "_out_" + str(output_unit)
                            edgeList.append({"data": {"id": count_edge, "source": unitInName, "target": unitOutName, "width": abs(float( layer.weight[output_unit][input_unit]))}})


        else:
            if len(name) > 0:
                nodeList.append({"data": {"id": name, "type": layer.__class__.__name__}})

    # ダミー入力を流してフックを発火させる
    _ = model(input_tensor)

    for edge in edges:
        if len(edge[0]) > 0 and len(edge[1]) > 0:
            count_edge += 1
            edgeList.append({"data": {"id": count_edge, "source": edge[0], "target": edge[1], "width": 1}})

    for edgeIdx in range(len(edgeList)):
        for nodeName in adjustNode:
            if edgeList[edgeIdx]["data"]["source"] == nodeName and "UNIT_" + nodeName not in edgeList[edgeIdx]["data"]["target"]:
                edgeList[edgeIdx]["data"]["source"] = nodeName + "_out"

    if len(existing_graphs) > 0:
        prevEdgeList = existing_graphs[-1]['graph']['edges']
        for edgeIdx in range(len(edgeList)):
            edgeData = {
                "data": {"id": edgeList[edgeIdx]['data']['id'], 
                "source": edgeList[edgeIdx]['data']['source'], 
                "target": edgeList[edgeIdx]['data']['target'],
                "width": edgeList[edgeIdx]['data']['width'] - prevEdgeList[edgeIdx]['data']['width']
            }}
#            edgeDiffList.append(edgeList[edgeIdx])
#            edgeDiffList[-1]['data']['width'] = edgeList[edgeIdx]['data']['width'] - prevEdgeList[edgeIdx]['data']['width']
            edgeDiffList.append(edgeData)
    else:
        edgeDiffList = edgeList


    
    json_graph = convert_to_json_format(nodeList, edgeList)
    json_graph_widthDiff = convert_to_json_format(nodeList, edgeDiffList)
    return json_graph, json_graph_widthDiff

