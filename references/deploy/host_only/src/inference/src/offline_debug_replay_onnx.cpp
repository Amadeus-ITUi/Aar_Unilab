// !!!! WARNING - STALE TOOL !!!!
// 本工具基于旧的 27 维 obs + Encoder + Policy 双模型架构 (135 D actor input,
// frame-major CSV, 7 terms 无 wing_vel), 与当前 lab_inference_node 使用的
// 145 D actor input / 单 lab_policy.mnn / 8 terms (含 wing_vel_obs) / term-major
// 展开完全不兼容。
//
// 使用前必须完整重写：CSV 列宽 29 维、历史 term-major 展开、单模型输入 145 维、
// 观测里加入 wing_vel 段。当前仅保留源码用于历史参考。请勿基于此工具的输出
// 得出"部署与训练一致/不一致"的结论。
//
// (原注释)
// 基于 ONNXRuntime 的离线回放调试节点：读取实机 CSV（obs + last_action），
// 复现 Encoder + Policy 推理，并对比「当前帧预测的 raw action」与「下一帧 obs 中记录的 last_action」。

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

struct FrameData {
    std::vector<float> obs27;      // obs_0..obs_26
    std::vector<float> last_act6;  // obs_21..obs_26
};

// 读取 CSV: time, obs_0..obs_26[, obs_27..obs_29, act_0..5...]
std::vector<FrameData> load_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开 CSV 文件: " + path);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV 文件为空或无法读取表头: " + path);
    }

    std::vector<FrameData> frames;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string val;
        FrameData f;
        f.obs27.reserve(27);
        f.last_act6.reserve(6);

        // 跳过 time
        std::getline(ss, val, ',');

        // 读 obs_0..obs_26
        for (int i = 0; i < 27; ++i) {
            if (std::getline(ss, val, ',')) {
                f.obs27.push_back(std::stof(val));
            } else {
                f.obs27.push_back(0.0f);
            }
        }

        // 将 obs_21..obs_26 视为 last_action
        for (int i = 0; i < 6; ++i) {
            int idx = 21 + i;
            if (idx < static_cast<int>(f.obs27.size())) {
                f.last_act6.push_back(f.obs27[idx]);
            } else {
                f.last_act6.push_back(0.0f);
            }
        }

        frames.push_back(std::move(f));
    }
    return frames;
}

// 使用 ONNXRuntime 跑单输入单输出模型
std::vector<float> run_onnx(
    Ort::Session& session,
    Ort::MemoryInfo& mem_info,
    const std::vector<float>& input_data
) {
    Ort::AllocatorWithDefaultOptions allocator;

    // 输入信息（旧版 C++ API 通过 GetInputNameAllocated 获取名称）
    if (session.GetInputCount() != 1) {
        throw std::runtime_error("ONNX 模型输入数量不是 1");
    }
    Ort::AllocatedStringPtr in_name_ptr = session.GetInputNameAllocated(0, allocator);
    std::string in_name(in_name_ptr.get());

    auto in_type_info = session.GetInputTypeInfo(0);
    auto in_tensor_info = in_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> in_shape = in_tensor_info.GetShape();

    int64_t total_elems = 1;
    for (auto& d : in_shape) {
        if (d < 0) d = 1;
        total_elems *= d;
    }
    if (static_cast<size_t>(total_elems) != input_data.size()) {
        throw std::runtime_error("输入尺寸不匹配: 模型期望 " +
                                 std::to_string(total_elems) + " 实际 " +
                                 std::to_string(input_data.size()));
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        in_shape.data(),
        in_shape.size()
    );

    // 输出信息
    if (session.GetOutputCount() != 1) {
        throw std::runtime_error("ONNX 模型输出数量不是 1");
    }
    Ort::AllocatedStringPtr out_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string out_name(out_name_ptr.get());

    const char* in_names[] = { in_name.c_str() };
    const char* out_names[] = { out_name.c_str() };

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        in_names,
        &input_tensor,
        1,
        out_names,
        1
    );

    float* out_data = outputs[0].GetTensorMutableData<float>();
    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> out_shape = out_info.GetShape();
    size_t out_elems = 1;
    for (auto d : out_shape) {
        out_elems *= d;
    }

    return std::vector<float>(out_data, out_data + out_elems);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "用法: offline_debug_replay_onnx <csv_path> <policy.onnx> <encoder.onnx>\n";
        return -1;
    }

    std::string csv_path     = argv[1];
    std::string policy_path  = argv[2];
    std::string encoder_path = argv[3];

    try {
        auto frames = load_csv(csv_path);
        if (frames.size() < 6) {
            std::cerr << "帧数过少: " << frames.size() << "，至少需要 6 帧。\n";
            return -1;
        }
        std::cout << "已加载 " << frames.size() << " 帧数据。\n";

        // 初始化 ONNXRuntime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "offline_debug_replay_onnx");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session encoder_sess(env, encoder_path.c_str(), opts);
        Ort::Session policy_sess(env, policy_path.c_str(), opts);
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        const int obs_dim = 27;
        const int history_len = 5;

        std::cout << std::fixed << std::setprecision(4);

        for (size_t f = 0; f < frames.size(); ++f) {
            if (f + 1 < static_cast<size_t>(history_len)) {
                continue;
            }

            // 1. 构建 Encoder 输入：5帧 x 27维 = 135
            std::vector<float> encoder_input(history_len * obs_dim, 0.0f);
            for (int i = 0; i < history_len; ++i) {
                size_t idx = f + 1 - history_len + i;  // [f-4..f]
                const auto& obs = frames[idx].obs27;
                for (int j = 0; j < obs_dim; ++j) {
                    encoder_input[i * obs_dim + j] =
                        (j < static_cast<int>(obs.size())) ? obs[j] : 0.0f;
                }
            }

            auto latent = run_onnx(encoder_sess, mem_info, encoder_input);
            if (latent.size() < 3) {
                throw std::runtime_error("Encoder 输出维度不足 3，实际: " +
                                         std::to_string(latent.size()));
            }

            // 2. 构建 Policy 输入：当前帧 27维 obs + 3维 latent
            const auto& cur = frames[f].obs27;
            std::vector<float> policy_input(30, 0.0f);
            for (int i = 0; i < obs_dim; ++i) {
                policy_input[i] = (i < static_cast<int>(cur.size())) ? cur[i] : 0.0f;
            }
            for (int i = 0; i < 3; ++i) {
                policy_input[obs_dim + i] = latent[i];
            }

            // 打印当前帧输入
            std::cout << "Frame " << f << " policy_input[0..29]: ";
            for (int i = 0; i < 30; ++i) {
                std::cout << policy_input[i];
                if (i != 29) std::cout << ", ";
            }
            std::cout << std::endl;

            // 3. Policy 推理
            auto act = run_onnx(policy_sess, mem_info, policy_input);
            if (act.size() < 6) {
                throw std::runtime_error("Policy 输出维度不足 6，实际: " +
                                         std::to_string(act.size()));
            }

            float pred_L = act[2];
            float pred_R = act[5];

            // 4. 对比：下一帧 obs_21..26 中记录的 last_action[2/5]
            float csv_next_L = 0.0f, csv_next_R = 0.0f;
            if (f + 1 < frames.size()) {
                const auto& next = frames[f + 1].obs27;
                if (next.size() > 23) csv_next_L = next[21 + 2];
                if (next.size() > 26) csv_next_R = next[21 + 5];
            }

            float obs_vel_L = (cur.size() > 20) ? cur[20] : 0.0f;
            float obs_vel_R = (cur.size() > 26) ? cur[26] : 0.0f;

            std::cout << "Frame " << f
                      << " | obs_vel_L=" << std::setw(8) << obs_vel_L
                      << " obs_vel_R=" << std::setw(8) << obs_vel_R
                      << " | pred_raw_act[2]=" << std::setw(8) << pred_L
                      << " pred_raw_act[5]=" << std::setw(8) << pred_R
                      << " | csv_next_last_obs[2]=" << std::setw(8) << csv_next_L
                      << " csv_next_last_obs[5]=" << std::setw(8) << csv_next_R
                      << std::endl;
        }

        std::cout << "ONNX 离线回放完成，共处理 " << frames.size() << " 帧。" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

