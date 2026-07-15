#!/bin/bash
# ============================================================
# v1.5 联合卸载优化 — 服务器部署验证脚本
# 用法: bash deploy_verify.sh
# ============================================================
set -e

echo "=============================================="
echo " v1.5 联合卸载优化 — 部署验证"
echo " $(date)"
echo "=============================================="

# ── 检查 Python 环境 ──
echo ""
echo "[1/5] Python environment..."
PYTHON="${PYTHON:-$(which python3 2>/dev/null || which python 2>/dev/null)}"
echo "  Python: $($PYTHON --version 2>&1)"
$PYTHON -c "import numpy; print('  numpy: ', numpy.__version__)" 2>/dev/null || { echo "ERROR: numpy not found"; exit 1; }
$PYTHON -c "import torch; print('  torch: ', torch.__version__)" 2>/dev/null || { echo "ERROR: torch not found"; exit 1; }
echo "  OK"

# ── 编译检查 ──
echo ""
echo "[2/5] Compile check..."
$PYTHON -m py_compile config.py || { echo "ERROR: config.py"; exit 1; }
$PYTHON -m py_compile environment/env.py || { echo "ERROR: environment/env.py"; exit 1; }
$PYTHON -m py_compile environment/uavs.py || { echo "ERROR: environment/uavs.py"; exit 1; }
$PYTHON -m py_compile environment/user_equipments.py || { echo "ERROR: environment/user_equipments.py"; exit 1; }
$PYTHON -m py_compile environment/comm_model.py || { echo "ERROR: environment/comm_model.py"; exit 1; }
$PYTHON -m py_compile train.py || { echo "ERROR: train.py"; exit 1; }
$PYTHON -m py_compile main.py || { echo "ERROR: main.py"; exit 1; }
echo "  OK — 7 files"

# ── 配置检查 ──
echo ""
echo "[3/5] Config validation..."
$PYTHON -c "
import config
print(f'  OFFLOAD_PIPELINE_MODE = {config.OFFLOAD_PIPELINE_MODE}')
print(f'  MBS_COMPUTING_CAPACITY = {config.MBS_COMPUTING_CAPACITY:.1e}')
print(f'  OFFLOAD_LATENCY_REF = {config.OFFLOAD_LATENCY_REF}')
print(f'  OFFLOAD_ENERGY_REF = {config.OFFLOAD_ENERGY_REF}')
print(f'  OFFLOAD_MAX_ITERATIONS = {config.OFFLOAD_MAX_ITERATIONS}')
assert config.OFFLOAD_PIPELINE_MODE in config.VALID_OFFLOAD_PIPELINE_MODES
print('  OK — config validation passed')
"

# ── Env 初始化 + 四模式冒烟 ──
echo ""
echo "[4/5] Four-mode smoke test..."

# 用独立种子避免模式间干扰
for MODE in legacy unified_fixed_targets iterative_latency iterative_joint; do
    $PYTHON -c "
import config, numpy as np, torch
config.OFFLOAD_PIPELINE_MODE = '$MODE'
config.VERIFY_TIMING = True
config.COLLECT_ROUTE_NORMALIZATION_STATS = False
torch.manual_seed(42)
np.random.seed(42)
config.SEED = 42

from environment.env import Env
env = Env()
env.reset()
actions = np.random.uniform(-1, 1, (config.NUM_UAVS, config.ACTION_DIM))
next_obs, rewards, metrics = env.step(actions)

# 检查基础
assert len(next_obs) == config.NUM_UAVS
assert len(rewards) == config.NUM_UAVS
assert all(np.isfinite(r) for r in rewards), 'NaN reward'
assert np.isfinite(metrics[0]), 'NaN latency'
assert np.isfinite(metrics[1]), 'NaN energy'

# 模式特定检查
if '$MODE' in ('iterative_latency', 'iterative_joint'):
    assert env._last_offload_converged, 'Not converged'
    assert env._j5_verified_service_steps > 0, 'J5 not called'

print(f'  $MODE: OK  reward={sum(rewards):.2f}  lat={metrics[0]:.0f}  eng={metrics[1]:.0f}')
" 2>&1 || { echo "  $MODE: FAILED"; exit 1; }
done

# 固定种子对比 legacy ↔ unified_fixed_targets 的 target
echo ""
$PYTHON -c "
import config, numpy as np, torch
SEED=42
torch.manual_seed(SEED); np.random.seed(SEED); config.SEED=SEED
from environment.env import Env

# legacy
config.OFFLOAD_PIPELINE_MODE='legacy'; config.VERIFY_TIMING=False
e1=Env(); e1.reset()
for u in e1.uavs: u.prepare_request_slot()
t1={p.ue.id:(p.target_idx,p.target_uav.id if p.target_uav else None) for p in e1._build_legacy_plans()}

# unified_fixed_targets
torch.manual_seed(SEED); np.random.seed(SEED); config.SEED=SEED
config.OFFLOAD_PIPELINE_MODE='unified_fixed_targets'
e2=Env(); e2.reset()
for u in e2.uavs: u.prepare_request_slot()
t2={p.ue.id:(p.target_idx,p.target_uav.id if p.target_uav else None) for p in e2._build_legacy_plans()}

assert t1==t2, f'Target mismatch: {sum(1 for k in t1 if t1[k]!=t2.get(k))} diffs'
print('  Target consistency: legacy == unified_fixed_targets — OK')
"

echo ""
echo "[5/5] Quick convergence & J5 check..."
$PYTHON -c "
import config, numpy as np, torch
config.OFFLOAD_PIPELINE_MODE='iterative_latency'
config.VERIFY_TIMING=True
config.COLLECT_ROUTE_NORMALIZATION_STATS=False
torch.manual_seed(42); np.random.seed(42); config.SEED=42
from environment.env import Env
env=Env()
for _ in range(50):
    env.reset()
    a=np.random.uniform(-1,1,(config.NUM_UAVS,config.ACTION_DIM))
    env.step(a)
nc=sum(1 for c in env._offload_convergence_history if not c)
avg_iter=np.mean(env._offload_iteration_history)
print(f'  50 steps: avg_iters={avg_iter:.1f}  non-converged={nc}  J5_count={env._j5_verified_service_steps}')
assert nc==0, f'Non-converged steps: {nc}'
assert env._j5_verified_service_steps>0, 'J5 not called'
print('  OK — 100% convergence, J5 verified')
"

# ── 汇总 ──
echo ""
echo "=============================================="
echo " ALL CHECKS PASSED"
echo "=============================================="
echo " Config:  MODE=legacy (default, safe for training)"
echo " To use:  set config.OFFLOAD_PIPELINE_MODE before import"
echo "   legacy                — original behavior"
echo "   unified_fixed_targets  — new estimator + executor"
echo "   iterative_latency      — coordinate descent (latency)"
echo "   iterative_joint        — coordinate descent (latency+energy)"
echo "=============================================="
