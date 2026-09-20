"""Reproduce the dated reward comparison, checking the owner configuration snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize_config_dir
from matplotlib.font_manager import FontProperties
from omegaconf import OmegaConf

from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.dr002.joystick import DR002JoystickEnv, RewardConfig
from unilab.envs.locomotion.pe03.config import load_config

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "conf/ppo")):
    WE = compose(config_name="config", overrides=["task=dr002_joystick_flat_we11/mujoco"])
PE = load_config(["+experiment=gait_fixed"])
W = OmegaConf.to_container(WE.reward, resolve=True)
Q = OmegaConf.to_container(PE.reward, resolve=True)
WS, PS = W["scales"], Q["scales"]
DT = float(WE.env.ctrl_dt)
PDT = 1 / PE.control.policy_hz
SIGMA = Q["sigma_negative"]
assert DT == PDT == 0.02
BLUE, ORANGE = "#2563a6", "#dc7025"
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": FONT.get_name(), "font.size": 10, "axes.unicode_minus": False})


def wvx(e):
    return np.minimum(
        WS["track_lin_vel_x"] * np.exp(-((e / W["track_lin_vel_x_std"]) ** 2)),
        W["track_lin_vel_x_term_clip"],
    )


def waux(e):
    return np.maximum(
        WS["track_lin_vel_x_enhance"]
        * (np.exp(-((e / W["track_lin_vel_x_enhance_std"]) ** 2)) - 1),
        -W["track_lin_vel_x_term_clip"],
    )


def wyaw(e):
    return np.minimum(
        WS["track_ang_vel_z"] * np.exp(-((e / W["tracking_sigma"]) ** 2)),
        W["track_ang_vel_z_term_clip"],
    )


def pvx(e):
    return PS["tracking_lin_vel"] * np.exp(-(e**2) / Q["tracking_sigma"])


def pyaw(e):
    return PS["tracking_ang_vel"] * np.exp(-(e**2) / Q["yaw_tracking_sigma"])


def penalty_w(name, squared_error, cap=1):
    return np.maximum(WS[name] * squared_error, -cap)


def check_we11_functions():
    """Compare plot expressions with bound production reward methods (no physics)."""
    env = object.__new__(DR002JoystickEnv)
    env._cfg = SimpleNamespace(ctrl_dt=DT)
    env._reward_cfg = RewardConfig(**W)
    x = np.linspace(0, 2, 129)
    n = len(x)
    ctx = RewardContext(
        info={"commands": np.zeros((n, 3))},
        linvel=np.zeros((n, 3)),
        gyro=np.zeros((n, 3)),
        dof_pos=np.zeros((n, 6)),
        num_envs=n,
        tracking_sigma=W["tracking_sigma"],
        base_height=x / 10,
        gravity=np.column_stack((np.sin(x), np.zeros(n), -np.cos(x))),
    )
    ctx.linvel[:, 0] = x
    ctx.linvel[:, 2] = x
    ctx.gyro[:, 2] = x
    ctx.gyro[:, 0] = x
    ctx.info["current_actions"] = np.column_stack((x, np.zeros((n, 5))))
    ctx.info["last_actions"] = np.zeros((n, 6))
    ctx.info["qacc"] = ctx.info["current_actions"] * 500
    checks = [
        ("track_lin_vel_x", env._reward_track_lin_vel_x, wvx(x)),
        ("track_lin_vel_x_enhance", env._reward_track_lin_vel_x_enhance, waux(x)),
        ("track_ang_vel_z", env._reward_track_ang_vel_z, wyaw(x)),
        ("lin_vel_z", env._reward_lin_vel_z_lingzu, penalty_w("lin_vel_z", x**2)),
        ("ang_vel_xy", env._reward_ang_vel_xy_lingzu, penalty_w("ang_vel_xy", x**2)),
        (
            "orientation",
            env._reward_orientation_lingzu,
            penalty_w("orientation", np.sin(x) ** 2, W["orientation_term_clip"]),
        ),
        (
            "base_height",
            env._reward_base_height_cmd,
            penalty_w("base_height", (ctx.base_height / W["base_height_std"]) ** 2, 4),
        ),
        ("action_rate_l2", env._reward_action_rate_lingzu, penalty_w("action_rate_l2", x**2)),
        (
            "joint_acc_l2",
            env._reward_joint_acc_l2,
            penalty_w("joint_acc_l2", (500 * x) ** 2),
        ),
    ]
    for name, method, expected in checks:
        np.testing.assert_allclose(method(ctx) * WS[name], expected, rtol=2e-6, atol=3e-7)
    return [name for name, _, _ in checks]


def finish(fig, name, title, subtitle):
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=10, color="#444444")
    for ax in fig.axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout(rect=(0, 0.015, 1, 0.915), h_pad=2.5, w_pad=2)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=165, facecolor="white")
    plt.close(fig)


def pair(ax, x, we, pe, xlabel, title, ylabel="加权项（乘 dt 前）"):
    ax.plot(x, we, color=BLUE, lw=2.4, label="WE11：直接进入加法总奖励")
    ax.plot(x, pe, color=ORANGE, lw=2.4, label="PE03：尚未经过指数聚合")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)


def tracking():
    e = np.linspace(0, 1.5, 600)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(e, np.exp(-((e / 0.25) ** 2)), color=BLUE, lw=2.4, label="WE11：exp(-误差² / 0.25²)")
    ax.plot(e, np.exp(-(e**2) / 0.25), color=ORANGE, lw=2.4, label="PE03：exp(-误差² / 0.25)")
    ax.set(
        xlabel="前向速度误差绝对值（m/s）",
        ylabel="未加权原始得分",
        title="A  同名参数 0.25，曲线宽度不同",
    )
    ax = axes[0, 1]
    pair(
        ax,
        e,
        wvx(e) + waux(e),
        pvx(e),
        "前向速度误差绝对值（m/s）",
        "B  前向速度：WE11 主项 + enhance 项",
    )
    ax.plot(e, wvx(e), color=BLUE, ls="--", alpha=0.6, label="WE11 主项（单独显示）")
    ax.plot(e, waux(e), color="#67835b", ls=":", lw=2, label="WE11 enhance（单独显示）")
    ax.axhline(0, color="#777777", lw=0.7)
    pair(
        axes[1, 0],
        e,
        wyaw(e),
        pyaw(e),
        "yaw 角速度误差绝对值（rad/s）",
        "C  yaw 跟踪：峰值和宽度都不同",
    )
    v = np.linspace(0, 0.25, 400)
    ax = axes[1, 1]
    ax.plot(v, np.maximum(-20 * v, -2), color=BLUE, lw=2.4, label="WE11：-min(20 × |速度|, 2)")
    ax.plot(v, np.zeros_like(v), color=ORANGE, lw=2.4, label="PE03：无对应的额外静止项")
    ax.set(
        xlabel="实际 |vx|（m/s）或 |yaw 角速度|（rad/s）",
        ylabel="额外加权项（乘 dt 前）",
        title="D  零命令时 WE11 另加两项惩罚",
    )
    finish(
        fig,
        "01_tracking",
        "速度跟踪：两者都有指数单项，但形状和组合不同",
        "当前配置；B/C 假设零命令专用惩罚未启用；PE03 的 vy 误差取 0；dt = 0.02 s",
    )


def penalties():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    h = np.linspace(0, 0.2, 400)
    pair(
        axes[0, 0],
        h * 100,
        penalty_w("base_height", (h / 0.05) ** 2, 4),
        -10 * h**2,
        "高度误差绝对值（cm）",
        "A  高度：WE11 除以 std²，PE03 不除",
    )
    deg = np.linspace(0, 90, 400)
    sq = np.sin(np.deg2rad(deg)) ** 2
    pair(
        axes[0, 1],
        deg,
        penalty_w("orientation", sq, 3),
        -5 * sq,
        "身体偏离竖直的倾角（°）",
        "B  姿态：WE11 在约 33.2° 后封顶",
    )
    v = np.linspace(0, 2, 400)
    pair(
        axes[0, 2],
        v,
        penalty_w("lin_vel_z", v**2),
        -0.02 * v**2,
        "机体 |vz|（m/s）",
        "C  竖直速度：当前权重相差 100 倍",
    )
    a = np.linspace(0, 300, 400)
    pair(
        axes[1, 0],
        a,
        penalty_w("joint_acc_l2", a**2),
        -2.5e-7 * a**2,
        "一个腿关节 |加速度|（rad/s²）",
        "D  关节加速度：WE11 单项封顶",
    )
    d = np.linspace(0, 4, 400)
    pair(
        axes[1, 1],
        d,
        penalty_w("action_rate_l2", d**2),
        -0.01 * d**2,
        "动作一阶差分的 L2 范数",
        "E  动作平滑：动作尺度不同，不能只比权重",
    )
    n = np.arange(8)
    ax = axes[1, 2]
    ax.plot(n, np.maximum(-10 * n, -10), "o-", color=BLUE, label="WE11：-min(10N, 10)")
    ax.plot(n, -5 * n, "o-", color=ORANGE, label="PE03：-5N，位于指数外")
    ax.set(
        xlabel="各自判定的接触部件数 N",
        ylabel="直接加到总奖励的项 / dt",
        title="F  碰撞：判定传感量和阈值也不同",
    )
    finish(
        fig,
        "02_penalties",
        "惩罚单项：WE11 单项裁剪；PE03 大部分交给总指数处理",
        "A–E：PE03 曲线是聚合前负项，不是最终扣分；所有图均使用当前实际权重",
    )


def aggregation():
    c = np.linspace(0, 0.2, 500)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(c, 0.03 - c, color=BLUE, lw=2.4, label="加法：0.03 - C")
    ax.plot(c, 0.03 * np.exp(-c / SIGMA), color=ORANGE, lw=2.4, label="指数：0.03 × exp(-C / 0.02)")
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set(
        xlabel="惩罚量 C（已乘 dt）",
        ylabel="每策略步总奖励",
        title="A  只比较聚合机制：假设相同 P = 0.03",
    )
    ax = axes[0, 1]
    dw = DT * 0.4 * ((0.05 / 0.05) ** 2 - (0.01 / 0.05) ** 2)
    dp = 0.03 * (np.exp(-10 * 0.01**2) - np.exp(-10 * 0.05**2)) * np.exp(-c / SIGMA)
    ax.semilogy(
        c, np.full_like(c, dw), color=BLUE, lw=2.4, label="WE11：高度改善的收益不依赖其他项"
    )
    ax.semilogy(c, dp, color=ORANGE, lw=2.4, label="PE03：高度改善的收益也被其他项衰减")
    ax.set(
        xlabel="其他惩罚 C_other（已乘 dt）",
        ylabel="高度误差 5 cm → 1 cm 的奖励增量",
        title="B  相同改善，在差状态下是否仍然有收益？",
    )
    e = np.linspace(0, 1, 500)
    colors = ["#167b63", "#3d78b4", "#b7791f", "#a74754"]
    for cost, color in zip([0, 0.02, 0.06, 0.1], colors, strict=True):
        axes[1, 0].plot(
            e,
            PDT * (pvx(e) + pyaw(0)) * np.exp(-cost / SIGMA),
            color=color,
            lw=2,
            label=f"其他惩罚 = {cost:.2f}",
        )
        axes[1, 1].plot(
            e,
            DT * (wvx(e) + waux(e) + wyaw(0)) - cost,
            color=color,
            lw=2,
            label=f"其他惩罚 = {cost:.2f}",
        )
    axes[1, 0].set(
        xlabel="前向速度误差绝对值（m/s）",
        ylabel="每策略步总奖励",
        title="C  PE03：其他惩罚越大，整条跟踪曲线越扁",
    )
    axes[1, 1].set(
        xlabel="前向速度误差绝对值（m/s）",
        ylabel="每策略步总奖励",
        title="D  WE11：其他惩罚使跟踪曲线整体下移",
    )
    finish(
        fig,
        "03_aggregation",
        "总奖励：加法保留各项差异，指数把多项误差耦合在一起",
        "曲线为解析切片，不是训练成绩；B–D 用当前参数、yaw 误差为 0、无碰撞；C/D 未启用零命令专用项",
    )


def main():
    checks = check_we11_functions()
    paths = [
        "conf/ppo/task/dr002_joystick_flat_we11/base.yaml",
        "conf/ppo/task/dr002_joystick_rough_we11/mujoco.yaml",
        "conf/pe03/task/pe03_gait_flat.yaml",
        "src/unilab/envs/locomotion/dr002/joystick.py",
        "src/unilab/envs/locomotion/dr002/rough.py",
        "src/unilab/envs/locomotion/common/rewards.py",
        "src/unilab/envs/locomotion/pe03/gait_env.py",
    ]
    snapshot = {
        "date": "2026-09-20",
        "we11_task": "dr002_joystick_flat_we11/mujoco",
        "pe03_experiment": "gait_fixed",
        "dt": DT,
        "we11_reward": W,
        "pe03_reward": Q,
        "we11_curves_checked_against_production_methods": checks,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
    }
    snapshot_path = OUT / "snapshot.json"
    if snapshot_path.exists():
        previous = json.loads(snapshot_path.read_text())
        for key in ("dt", "we11_reward", "pe03_reward", "source_sha256"):
            if previous[key] != snapshot[key]:
                raise ValueError(
                    "Source/configuration changed since this dated comparison. "
                    "Review formulas, captions and report before replacing the snapshot."
                )
    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    tracking()
    penalties()
    aggregation()
    print(f"Validated {len(checks)} WE11 reward expressions; wrote 3 PNG/SVG/PDF figure sets.")


if __name__ == "__main__":
    main()
