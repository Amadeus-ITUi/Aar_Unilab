# WE11 regression baseline

On 2026-09-15, the imported dirty worktree and the new independent repository
both produced the same two failures:

- `test_we11_base_config_owns_latest_network_observation_and_force_contract`:
  the test expects delay minimum 4 while the preserved config resolves to 2.
- `test_height_reward_clip_keeps_signal_through_five_centimetres`: the test
  expects the third reward value to clip to 4 while the preserved implementation
  returns 5.76.

The new repository's complete non-slow result before any baseline correction
was **195 passed, 2 failed**. These expectations are intentionally not changed
during migration because doing so would alter the user's tuned WE11 behavior or
declare a new expected value without a training decision.
