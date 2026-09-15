// !!!! WARNING - STALE TOOL !!!!
// 本工具基于旧的 27 维 obs + Encoder + Policy 双模型架构 (135 D actor input,
// 无 wing_vel 段), 与当前 lab_inference_node 的 145 D 单模型 / 8 terms 不兼容。
// 使用前必须完整重写为 29 维 obs / term-major / 单模型 145 维输入。
//
// (原注释) 离线推理一致性检查工具：基于 CSV 中的 27 维 obs，重现 Encoder + Policy 推理过程，
// 打印轮子相关观测与策略输出，用于分析“轮子飞转”等异常现象。

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <deque>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

// 定义模型上下文，模拟实机运行环境
struct ModelContext {
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
    MNN::Tensor* input_tensor = nullptr;
    MNN::Tensor* output_tensor = nullptr;
};

// 加载 MNN 模型并初始化 Session
void setup_mnn(std::unique_ptr<ModelContext>& ctx, const std::string& model_path) {
    ctx = std::make_unique<ModelContext>();
    ctx->net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!ctx->net) {
        throw std::runtime_error("Failed to create MNN Interpreter from file: " + model_path);
    }
    MNN::ScheduleConfig config;
    config.numThread = 1; // 调试时使用单线程确保结果确定性
    ctx->session = ctx->net->createSession(config);
    if (!ctx->session) {
        throw std::runtime_error("Failed to create MNN session for file: " + model_path);
    }
    ctx->input_tensor = ctx->net->getSessionInput(ctx->session, nullptr);
    ctx->output_tensor = ctx->net->getSessionOutput(ctx->session, nullptr);
    if (!ctx->input_tensor || !ctx->output_tensor) {
        throw std::runtime_error("Failed to get input/output tensor for file: " + model_path);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "用法: offline_inference_check <csv_path> <policy_mnn> <encoder_mnn>" << std::endl;
        return -1;
    }

    std::string csv_path = argv[1];
    std::unique_ptr<ModelContext> policy_ctx, encoder_ctx;

    try {
        std::cout << "正在载入模型..." << std::endl;
        setup_mnn(policy_ctx, argv[2]);
        setup_mnn(encoder_ctx, argv[3]);
    } catch (const std::exception& e) {
        std::cerr << "模型加载失败: " << e.what() << std::endl;
        return -1;
    }

    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << csv_path << std::endl;
        return -1;
    }

    std::string line;
    // 跳过第一行表头
    if (!std::getline(file, line)) {
        std::cerr << "CSV 文件为空或无法读取表头: " << csv_path << std::endl;
        return -1;
    }

    std::deque<std::vector<float>> history_buffer;
    const int history_len = 5;
    const int obs_dim = 27;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n开始分析... (每20帧采样一次)\n" << std::endl;
    std::cout << "帧号  |  左轮速度(obs20) | 右轮速度(obs26) | 策略输出:左轮(Act2) | 策略输出:右轮(Act5)" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    int frame_count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string val;
        std::vector<float> current_obs;
        current_obs.reserve(obs_dim);

        // 1. 解析 CSV：跳过第一列时间戳，读取后 27 列
        std::getline(ss, val, ','); 
        for (int i = 0; i < obs_dim; ++i) {
            if (std::getline(ss, val, ',')) {
                current_obs.push_back(std::stof(val));
            } else {
                current_obs.push_back(0.0f);
            }
        }

        // 2. 更新历史滑动窗口
        history_buffer.push_back(current_obs);
        if (history_buffer.size() > history_len) {
            history_buffer.pop_front();
        }

        // 等待窗口填满后开始推理
        if (history_buffer.size() < history_len) {
            ++frame_count;
            continue;
        }

        // 3. 执行 Encoder 推理 (重建 Latents)
        float* encoder_in = encoder_ctx->input_tensor->host<float>();
        for (int i = 0; i < history_len; ++i) {
            for (int j = 0; j < obs_dim; ++j) {
                // 当前实现：history_buffer[0] 是最老帧，[4] 是最新帧
                // 若训练端为 [t, t-1, ...] 顺序，可将 i 改为 (history_len - 1 - i)
                encoder_in[i * obs_dim + j] = history_buffer[i][j];
            }
        }
        encoder_ctx->net->runSession(encoder_ctx->session);
        float* latent_out = encoder_ctx->output_tensor->host<float>();

        // 4. 执行 Policy 推理 (27维原生观测 + 3维 Latent)
        float* policy_in = policy_ctx->input_tensor->host<float>();
        for (int i = 0; i < obs_dim; ++i) {
            policy_in[i] = current_obs[i];
        }
        for (int i = 0; i < 3; ++i) {
            policy_in[obs_dim + i] = latent_out[i];
        }

        policy_ctx->net->runSession(policy_ctx->session);
        float* action_out = policy_ctx->output_tensor->host<float>();

        // 5. 打印对比结果（每 20 帧打印一次）
        if (frame_count % 20 == 0) {
            float left_wheel_vel  = current_obs.size() > 20 ? current_obs[20] : 0.0f;
            float right_wheel_vel = current_obs.size() > 26 ? current_obs[26] : 0.0f;
            float act_left_wheel  = action_out[2];
            float act_right_wheel = action_out[5];

            std::cout << std::setw(5) << frame_count << " | "
                      << std::setw(16) << left_wheel_vel << " | "
                      << std::setw(16) << right_wheel_vel << " | "
                      << std::setw(18) << act_left_wheel << " | "
                      << std::setw(18) << act_right_wheel << std::endl;
        }

        ++frame_count;
    }

    std::cout << "\n分析结束。总计处理 " << frame_count << " 帧。" << std::endl;
    return 0;
}

