"""Exercise the complete v4 export in an isolated Torch/ONNX subprocess."""

import subprocess
import sys

from unilab.envs.locomotion.pe03.config import ROOT


def test_v4_export_relocation_and_live_commands(tmp_path):
    script = tmp_path / "check_export.py"
    script.write_text("""
import sys
import json
import shutil
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
import torch
import onnxruntime as ort
from unilab.envs.locomotion.pe03.config import load_config
from unilab.envs.locomotion.pe03.play_env import make_play_env
from unilab.algos.torch.pe03.runner import PE03Runner
from unilab.adapters.pe03_ppo import load_policy
from scripts.train_pe03 import export_release
cfg=load_config(['+experiment=gait_fixed','algo.num_envs=2','algo.num_steps_per_env=2',
 'algo.max_iterations=1','algo.num_learning_epochs=1','algo.num_mini_batches=1',
 'training.device=cpu','training.logger=none','training.mujoco_threads=1','training.evaluation_interval=0'])
r=PE03Runner(cfg,Path('run'))
try: checkpoint=r.learn()
finally:r.close()
release=export_release(checkpoint,'gait-test')
shutil.move(release,Path('relocated'))
policy=load_policy(checkpoint)
session=ort.InferenceSession('relocated/policy.onnx',providers=['CPUExecutionProvider'])
manifest=json.loads(Path('relocated/deployment_manifest.json').read_text())
runtime=json.loads((Path('relocated')/manifest['artifacts']['pe03_runtime_path']).read_text())
assert runtime['schema']=='pe03.runtime.v4.joint-limits.v1'
np.testing.assert_allclose(runtime['joint_target_limits'],
 [[-.20,1.50],[-1.44,0],[-2.02,0],[-1.50,.20],[0,1.44],[0,2.02]])
env=make_play_env(policy.config,Path('relocated')/manifest['artifacts']['scene_path'])
try:
 observation=env.reset()
 for index in range(30):
  if index==3: env.config.play.gait=[2.5,0.65,0.05]
  command=env.set_command(np.array([.2,-.1,.3]))
  inputs=dict(observation_history=observation.actor[None],observation=observation.actor[None,-38:],command=command[None])
  action=session.run(['action'],inputs)[0]
  with torch.no_grad(): expected=policy.action_mean(*[torch.from_numpy(v) for v in inputs.values()]).numpy()
  np.testing.assert_allclose(action,expected,atol=1e-6,rtol=1e-5)
  observation,reward,_,_=env.step(action[0])
  assert np.isfinite(reward)
 np.testing.assert_allclose(env.vector.gaits[0],[2.5,.65,.05],atol=1e-7)
finally:env.close()

from tools.compare_pe03_sim2sim import python_trajectory, cpp_trajectory
import onnx
from onnx import helper, numpy_helper
release_path=Path('relocated')
binary=Path(sys.argv[1])/'sim2sim/build/aar_sim2sim'
for label,constant in [('policy',None),('upper',1000.),('lower',-1000.)]:
 if constant is not None:
  model=onnx.load(release_path/'policy.onnx')
  del model.graph.node[:]
  del model.graph.initializer[:]
  model.graph.node.append(helper.make_node('Constant',[],['action'],
   value=numpy_helper.from_array(np.full((1,6),constant,dtype=np.float32))))
  onnx.save(model,release_path/'policy.onnx')
  np.save(release_path/'golden_outputs/action.npy',np.full((1,6),constant,dtype=np.float32))
 for steps in (1,10,100):
  expected=python_trajectory(release_path,steps)
  actual=cpp_trajectory(binary,release_path,steps,Path(f'telemetry-{label}-{steps}'))
  for a,b in zip(expected,actual,strict=True):
   assert a.keys()==b.keys()
   np.testing.assert_allclose([a[k] for k in a],[b[k] for k in a],rtol=0,atol=2e-5)
# Old self-contained releases still select the original, unclipped protocol.
from omegaconf import OmegaConf
import json
runtime_path=release_path/'robot/pe03_runtime.json'
runtime=json.loads(runtime_path.read_text())
runtime['schema']=runtime['schema'].removesuffix('.joint-limits.v1')
del runtime['joint_target_limits']
runtime_path.write_text(json.dumps(runtime))
old_config=OmegaConf.load(release_path/'runtime_config.yaml')
del old_config.control.clip_joint_targets
OmegaConf.save(old_config,release_path/'runtime_config.yaml')
expected=python_trajectory(release_path,10)
actual=cpp_trajectory(binary,release_path,10,Path('telemetry-legacy'))
for a,b in zip(expected,actual,strict=True):
 np.testing.assert_allclose([a[k] for k in a],[b[k] for k in a],rtol=0,atol=2e-5)
""")
    subprocess.run([sys.executable, str(script), str(ROOT)], cwd=tmp_path, check=True, timeout=180)
