#include <torch/script.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

namespace py = pybind11;
using json = nlohmann::json;

std::string analyze_model(const std::string& model_path, const std::string& output_json) {
    json model_info;
    try {
        torch::jit::script::Module module = torch::jit::load(model_path);
        for (const auto& param : module.named_parameters()) {
            model_info[param.name]["shape"] = param.value.sizes().vec();
        }
        std::ofstream output_file(output_json);
        output_file << model_info.dump(4);
        return "✅ JSON 書き込み完了: " + output_json;
    } catch (const c10::Error& e) {
        return "❌ モデルのロードに失敗しました";
    }
}

// Python から呼び出せるようにする
PYBIND11_MODULE(graph_analysis, m) {
    m.def("analyze_model", &analyze_model, "PyTorch モデルを解析し JSON に出力");
}

