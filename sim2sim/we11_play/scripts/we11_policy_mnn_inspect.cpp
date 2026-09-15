#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: we11_policy_mnn_inspect MODEL.mnn\n";
    return 2;
  }

  std::shared_ptr<MNN::Interpreter> net(
      MNN::Interpreter::createFromFile(argv[1]));
  if (!net) {
    std::cerr << "failed to load MNN model: " << argv[1] << "\n";
    return 3;
  }

  MNN::ScheduleConfig config;
  config.numThread = 1;
  MNN::Session* session = net->createSession(config);
  if (!session) {
    std::cerr << "failed to create MNN session\n";
    return 4;
  }

  MNN::Tensor* input = net->getSessionInput(session, nullptr);
  MNN::Tensor* output = net->getSessionOutput(session, nullptr);
  if (!input || !output) {
    std::cerr << "failed to obtain the model input/output tensors\n";
    net->releaseSession(session);
    return 5;
  }

  const int input_dim = input->elementSize();
  const int output_dim = output->elementSize();
  MNN::Tensor input_host(input, MNN::Tensor::CAFFE);
  std::fill_n(input_host.host<float>(), input_dim, 0.0F);
  input->copyFromHostTensor(&input_host);

  if (net->runSession(session) != MNN::NO_ERROR) {
    std::cerr << "MNN inference failed\n";
    net->releaseSession(session);
    return 6;
  }

  MNN::Tensor output_host(output, MNN::Tensor::CAFFE);
  output->copyToHostTensor(&output_host);
  bool finite = true;
  for (int i = 0; i < output_dim; ++i) {
    finite = finite && std::isfinite(output_host.host<float>()[i]);
  }

  std::cout << "input_dim=" << input_dim << " output_dim=" << output_dim
            << " finite_output=" << (finite ? "true" : "false") << "\n";
  net->releaseSession(session);

  if (input_dim != 145 || output_dim != 6 || !finite) {
    std::cerr << "expected a finite WE11 145D -> 6D model\n";
    return 7;
  }
  return 0;
}
