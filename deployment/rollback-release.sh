#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEPLOY_WAIT_TIMEOUT_SECONDS="${DEPLOY_WAIT_TIMEOUT_SECONDS:-360}"
ROLLBACK_API_IMAGE="${1:-}"

cd "$PROJECT_ROOT"

run_compose() {
  DOG_AGENT_API_IMAGE="$ROLLBACK_API_IMAGE" docker compose \
    --env-file "$ENV_FILE" \
    -f compose.yaml \
    -f compose.release.yaml \
    -f compose.proxy.yaml \
    "$@"
}

print_failure_context() {
  run_compose ps --all || true
  run_compose logs --tail 100 api nginx || true
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "缺少回滚命令: $command_name" >&2
    exit 1
  fi
}

if [[ -z "$ROLLBACK_API_IMAGE" ]]; then
  echo "用法: bash deployment/rollback-release.sh <固定 GHCR 镜像>" >&2
  echo "示例: bash deployment/rollback-release.sh ghcr.io/owner/dog-agent-api:1.21.0" >&2
  exit 2
fi

if [[ ! "$ROLLBACK_API_IMAGE" =~ ^ghcr\.io/[[:alnum:]._-]+/[[:alnum:]._/-]+:[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "回滚镜像必须是带三段固定版本的 GHCR 地址，禁止 latest。" >&2
  echo "收到: $ROLLBACK_API_IMAGE" >&2
  exit 2
fi

require_command docker
require_command "$PYTHON_BIN"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "缺少私有环境文件: $ENV_FILE" >&2
  exit 1
fi

echo "[1/5] 回滚前校验旧镜像与 API 最低安全配置。"
"$PYTHON_BIN" -m scripts.deployment.validate_v122_release_env \
  --env-file "$ENV_FILE" \
  --image-override "$ROLLBACK_API_IMAGE"

if ! docker info >/dev/null 2>&1; then
  echo "Docker Engine 不可用，请先启动 Docker 服务。" >&2
  exit 1
fi

trap print_failure_context ERR

echo "[2/5] 校验旧镜像覆盖后的 Compose 配置。"
run_compose config --quiet

echo "[3/5] 拉取指定的旧版 API 镜像。"
run_compose pull api

echo "[4/5] 使用旧镜像重建 API，并等待代理栈恢复健康。"
run_compose up \
  --detach \
  --remove-orphans \
  --wait \
  --wait-timeout "$DEPLOY_WAIT_TIMEOUT_SECONDS"

echo "[5/5] 重新验证 Nginx 入口和 FastAPI 端口隔离。"
"$PYTHON_BIN" -m scripts.deployment.verify_v122_proxy_deployment

trap - ERR
echo "Dog Agent 已回滚到: $ROLLBACK_API_IMAGE"
echo "请同步修改 .env 中的 DOG_AGENT_API_IMAGE，保证下次重启继续使用该版本。"
