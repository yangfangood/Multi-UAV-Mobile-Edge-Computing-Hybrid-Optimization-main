#!/bin/bash
# ============================================================
# v1.5 同步到服务器
# 用法：先修改下方 SERVER / PATH，然后 bash sync_to_server.sh
# ============================================================
set -e

# ── 修改这里 ──
SERVER="user@your-server-ip"
REMOTE_PATH="/path/to/Multi-UAV-Mobile-Edge-Computing-Hybrid-Optimization-main"
# ───────────────

FILES=(
    "config.py"
    "environment/env.py"
    "environment/uavs.py"
    "deploy_verify.sh"
)

echo "=============================================="
echo " Syncing v1.5 to server"
echo " Target: ${SERVER}:${REMOTE_PATH}"
echo "=============================================="

for f in "${FILES[@]}"; do
    echo "  Copying $f ..."
    scp "$f" "${SERVER}:${REMOTE_PATH}/$f"
done

echo ""
echo "Upload complete. Running deploy_verify on server..."
ssh "${SERVER}" "cd ${REMOTE_PATH} && bash deploy_verify.sh"

echo ""
echo "=============================================="
echo " Sync complete."
echo "=============================================="
