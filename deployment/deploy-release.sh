#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEPLOY_WAIT_TIMEOUT_SECONDS="${DEPLOY_WAIT_TIMEOUT_SECONDS:-360}"

cd "$PROJECT_ROOT"

compose_command=(
  docker compose
  --env-file "$ENV_FILE"
  -f compose.yaml
  -f compose.release.yaml
  -f compose.proxy.yaml
)

print_failure_context() {
  "${compose_command[@]}" ps --all || true
  "${compose_command[@]}" logs --tail 100 api nginx || true
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "缺少部署命令: $command_name" >&2
    exit 1
  fi
}

trap print_failure_context ERR

require_command docker
require_command "$PYTHON_BIN"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "缺少私有环境文件: $ENV_FILE" >&2
  echo "请根据 .env.example 创建 .env，并填写真实密钥与镜像版本。" >&2
  exit 1
fi

echo "[1/5] 部署前校验固定镜像与 API 最低安全配置。"
"$PYTHON_BIN" -m scripts.deployment.validate_v122_release_env \
  --env-file "$ENV_FILE"

if ! docker info >/dev/null 2>&1; then
  echo "Docker Engine 不可用，请先启动 Docker 服务。" >&2
  exit 1
fi

echo "[2/5] 校验三层 Docker Compose 配置。"
"${compose_command[@]}" config --quiet

echo "[3/5] 拉取固定版本 API 与 Nginx 镜像。"
"${compose_command[@]}" pull

echo "[4/5] 启动代理栈并等待健康检查。"
"${compose_command[@]}" up \
  --detach \
  --remove-orphans \
  --wait \
  --wait-timeout "$DEPLOY_WAIT_TIMEOUT_SECONDS"

echo "[5/5] 验证 Nginx 入口和 FastAPI 端口隔离。"
"$PYTHON_BIN" -m scripts.deployment.verify_v122_proxy_deployment

trap - ERR
echo "Dog Agent V1.22 发布镜像代理部署完成。"
