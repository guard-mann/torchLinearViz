import torch
import json

def convert_to_json_format(nodes):
  nodeDictList = []
  edgeDictList = []
  nodeIds = []
  count = 0
  for i, n in enumerate(nodes):
    nodeDictList.append({
        "data": {"id": n["outputs"][0], "type": n["kind"],}
    })
    nodeIds.append(n["outputs"][0])
    
  for i, n in enumerate(nodes):
    for idx in range(len(n["inputs"])):
        if n["inputs"][idx] in nodeIds:
            count += 1
            edgeDictList.append({
                "data": {
                    "id": f"e{str(count)}",
                    "source": n["inputs"][idx],
                    "target": n["outputs"][0],
                    "width": 1
                }
            })
  return {"nodes": nodeDictList, "edges": edgeDictList}

def extract_nodes(graph):
    nodes = []
    edges = []
    for node in graph.nodes():
      print(f"---\nkind: {node.kind()}, \ninputs: {[i.debugName() for i in node.inputs()]}, \noutputs: {[o.debugName() for o in node.outputs()]}")
      nodes.append({
        "kind": node.kind(),                  # ノードの種類 (例: aten::addmm)
        "inputs": [i.debugName() for i in node.inputs()],  # 入力ノード
        "outputs": [o.debugName() for o in node.outputs()] # 出力ノード
      })

    
    
    return nodes

def json_export(graph):
    with open('output.csv', 'w') as f:
      json.dump(graph, f, indent=4)


def extract_graph(model, input_tensor):
    traced_model = torch.jit.trace(model, input_tensor)
    graph = traced_model.graph
    node_info = extract_nodes(graph)
    json_graph = convert_to_json_format(node_info)
    json_export(json_graph)
