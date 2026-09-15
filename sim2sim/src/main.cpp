#include "aar/deployment_contract.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: aar_contract_check deployment_manifest.json\n";
    return 2;
  }
  try {
    const auto contract = aar::load_contract(argv[1]);
    std::cout << contract.schema << " robot=" << contract.robot_id
              << " task=" << contract.task_id << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
