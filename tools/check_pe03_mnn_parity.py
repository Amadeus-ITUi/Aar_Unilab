"""Read-only MNN/ONNX golden-case check, runnable on the Raspberry Pi.

Uses no ROS, serial device, robot service, or control command.
"""

import argparse
import json
from pathlib import Path

import MNN
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    cases = np.load(args.candidate / "parity_cases.npz", allow_pickle=False)
    interpreter = MNN.Interpreter(str(args.candidate / "policy.mnn"))
    session = interpreter.createSession({"numThread": 1, "backend": "CPU"})
    inputs = interpreter.getSessionInputAll(session)
    names = ("observation_history", "observation", "command")
    if set(inputs) != set(names):
        raise ValueError(f"MNN input contract mismatch: {list(inputs)}")
    outputs = []
    for i in range(len(cases["expected"])):
        for name in names:
            values = np.ascontiguousarray(cases[name][i : i + 1], dtype=np.float32)
            if tuple(inputs[name].getShape()) != values.shape:
                raise ValueError(f"MNN shape mismatch: {name}")
            tensor = MNN.Tensor(
                values.shape, MNN.Halide_Type_Float, values, MNN.Tensor_DimensionType_Caffe
            )
            inputs[name].copyFrom(tensor)
        interpreter.runSession(session)
        output = interpreter.getSessionOutput(session, "action")
        host = MNN.Tensor(
            output.getShape(),
            MNN.Halide_Type_Float,
            np.zeros(output.getShape(), dtype=np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        output.copyToHostTensor(host)
        outputs.append(np.asarray(host.getData(), dtype=np.float32).reshape(1, 6))
    actual = np.concatenate(outputs)
    np.testing.assert_allclose(actual, cases["expected"], atol=1e-4, rtol=1e-4)
    report = dict(
        passed=True,
        cases=len(actual),
        max_abs_action_error=float(np.max(abs(actual - cases["expected"]))),
        rmse_action=float(np.sqrt(np.mean((actual - cases["expected"]) ** 2))),
        threads=1,
        hardware_commands_sent=False,
    )
    (args.candidate / "mnn_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
