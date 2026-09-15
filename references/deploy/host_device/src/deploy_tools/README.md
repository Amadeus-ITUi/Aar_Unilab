# deploy_tools

ROS 2 operational tools for the Walking Eagle deployment workspace.

The active hardware backend is ESD-Link. See
`docs/ESD_LINK_OPERATOR_GUIDE.md` in the workspace for the current startup,
sweep, zeroing and emergency-disable entry points.

- `scripts/can`: legacy direct SocketCAN inspection tools, not used by ESD-Link
- `scripts/diagnostics`: read-only ROS diagnostics and runtime recorders
- `scripts/control`: explicit operator-invoked runtime parameter tools
- `scripts/hardware`: Raspberry Pi GPIO peripheral test tools
- `scripts/analysis`: offline CSV plotting and comparison tools

After building the workspace, each script is available through
`ros2 run deploy_tools <tool_name>`. Scripts that can transmit CAN or
change runtime parameters keep their existing explicit command-line behavior;
installing this package does not start any node or motor process.

The installed executable name omits the `.py` suffix. For example:

```bash
ros2 run deploy_tools read_motor_status --help
ros2 run deploy_tools plot_policy_pair_tracking --help
ros2 run deploy_tools emergency_disable_motors --help
ros2 run deploy_tools rgb_led_cycle --help
```

`rgb_led_cycle` 使用 BCM GPIO 编号，默认驱动共阳极 RGB LED：绿色接
GPIO17、蓝色接 GPIO27、红色接 GPIO22。每个颜色引脚必须串联独立的限流
电阻（建议从 220–330 Ω 起步），公共 `V` 引脚接 3.3 V。按 `Ctrl+C` 退出时
程序会熄灭 LED 并释放 GPIO。

可复用接口位于 `scripts/hardware/rgb_led.py`。RGB 分量范围为 0–255，亮度
范围为 0.0–1.0；`solid()` 常亮，`blink()` 在后台闪烁，`interval` 表示亮和
灭各自持续的秒数，`off()` 熄灭，`close()` 熄灭并释放 GPIO：

```python
from rgb_led import RGBLed

with RGBLed(red_pin=22, green_pin=17, blue_pin=27) as led:
    led.solid((255, 80, 0), brightness=0.4)
    led.blink((0, 0, 255), brightness=0.6, interval=0.2, count=5)
```

`emergency_disable_motors` 是旧 SocketCAN 后端工具。ESD-Link 模式必须先停止
bridge，再运行 `esd_link_emergency_disable`；它独占冻结固件的稳定串口路径并
等待整体失能确认。急停和切断动力电源仍是最终安全保障。

Tool categories describe ownership, not safety level. In particular, scripts in
`scripts/can` may transmit CAN frames when their existing command-line options
request it. Inspect each script's `--help` before using it on powered hardware.
