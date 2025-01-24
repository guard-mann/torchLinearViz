import torch
import json

def convert_to_json_format(nodes):
  nodeDictList = []
  for i, n in enumerate(nodes):
    nodeDictList.append({
        "id": n["outputs"][0],
        "kind": n["kind"],
        "inputs": n["inputs"]
    })
  return {"nodes": nodeDictList}

def attatch_node(graph):
    nodes = []
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
    node_info = attatch_node(graph)
    json_graph = convert_to_json_format(node_info)
    json_export(json_graph)
