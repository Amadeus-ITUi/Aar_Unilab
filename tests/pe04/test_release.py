"""Run Torch/ONNX/native release parity in an isolated process."""

import subprocess
import sys

from unilab.envs.locomotion.pe04.config import ROOT


def test_relocatable_export_and_cpp_runtime(tmp_path):
    script = tmp_path / "release_check.py"
    script.write_text("""
import sys, json, shutil, subprocess
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy as np
import torch
import onnxruntime as ort
from unilab.envs.locomotion.pe04.config import load_config
from unilab.envs.locomotion.pe04.play_env import make_play_env
from unilab.algos.torch.pe04.runner import PE04Runner
from unilab.adapters.pe04_ppo import load_policy
from scripts.train_pe04 import export_release
cfg=load_config(['algo.num_envs=2','algo.num_steps_per_env=4','algo.max_iterations=1',
 'algo.num_learning_epochs=1','algo.num_mini_batches=1','training.device=cpu',
 'training.logger=none','training.evaluation_interval=0','training.mujoco_threads=1',
 'play.delay_ms=25'])
r=PE04Runner(cfg,Path('run'))
try: checkpoint=r.learn()
finally: r.close()
release=export_release(checkpoint,'test')
shutil.move(release,'relocated')
release=Path('relocated')
manifest=json.loads((release/'deployment_manifest.json').read_text())
assert manifest['robot']['id']=='pe04'
assert manifest['control']['command_delay_steps']==10
policy=load_policy(checkpoint)
session=ort.InferenceSession(str(release/'policy.onnx'),providers=['CPUExecutionProvider'])
env=make_play_env(policy.config,release/manifest['artifacts']['scene_path'])
try:
 obs=env.reset()
 for _ in range(10):
  command=env.set_command(np.array([.2,-.1,.3]))[None]
  inputs=dict(observation_history=obs.actor[None],observation=obs.actor[None,-30:],command=command)
  output=session.run(['action'],inputs)[0]
  with torch.no_grad(): expected=policy.action_mean(*[torch.from_numpy(v) for v in inputs.values()]).numpy()
  np.testing.assert_allclose(output,expected,atol=1e-6,rtol=1e-5)
  obs,_,_,_=env.step(output[0])
 rows=[]
 obs=env.reset()
 for step in range(60):
  row=np.r_[obs.actor,obs.actor[-30:]]
  action=np.array([1000 if step==3 else .2*np.sin(step+i) for i in range(6)],np.float32)
  obs,_,_,_=env.step(action)
  rows.append(np.r_[row,env.data.ctrl])
 binary=Path(sys.argv[1])/'sim2sim/build/aar_pe04_runtime_check'
 assert binary.is_file(), 'Build sim2sim with the supported installation script first'
 subprocess.run([str(binary),str(release),'cpp.txt'],check=True)
 actual=np.loadtxt('cpp.txt')
 np.testing.assert_allclose(actual,np.array(rows),rtol=1e-5,atol=2e-5)
finally: env.close()
""")
    subprocess.run([sys.executable, str(script), str(ROOT)], cwd=tmp_path, check=True, timeout=120)
