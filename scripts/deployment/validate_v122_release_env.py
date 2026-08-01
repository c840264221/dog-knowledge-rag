"""校验 V1.22 生产部署使用的私有环境配置。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Mapping


GHCR_VERSIONED_IMAGE_PATTERN = re.compile(
    r"^ghcr\.io/[a-z0-9._-]+/[a-z0-9._/-]+:[0-9]+\.[0-9]+\.[0-9]+$",
    re.IGNORECASE,
)
PLACEHOLDER_API_KEYS = {
    "change-me",
    "changeme",
    "replace-me",
    "your-api-key",
}


def load_env_file(path: Path) -> dict[str, str]:
    """
    读取部署使用的 dotenv 环境文件。

    功能：
        解析 ``KEY=VALUE`` 形式的配置，兼容可选 ``export`` 前缀和成对引号；
        忽略空行及注释，不执行变量替换，也不会把配置值打印到日志。

    参数含义：
        path:
            需要检查的私有 ``.env`` 文件路径。

    返回值含义：
        dict[str, str]:
            环境变量名称到原始字符串值的映射。
    """

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(
                f"环境文件第 {line_number} 行不是 KEY=VALUE 格式"
            )
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"环境文件第 {line_number} 行缺少变量名")
        values[key] = _normalize_env_value(raw_value)
    return values


def _normalize_env_value(raw_value: str) -> str:
    """
    规范化 dotenv 中单个变量的文本值。

    功能：
        去除值两侧空白和成对单引号或双引号；未加引号时，移除由空格引出的
        行尾注释，避免把说明文字误认为配置值。

    参数含义：
        raw_value:
            等号右侧尚未处理的文本。

    返回值含义：
        str:
            可供生产配置规则检查的规范化字符串。
    """

    value = raw_value.strip()
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {'"', "'"}
    ):
        return value[1:-1]
    comment_index = value.find(" #")
    if comment_index >= 0:
        value = value[:comment_index]
    return value.strip()


def validate_release_environment(
    values: Mapping[str, str],
    *,
    image_override: str | None = None,
) -> list[str]:
    """
    检查生产发布必须满足的配置安全条件。

    功能：
        要求使用带三段版本号的 GHCR 镜像、启用 API Key 和限流、提供足够
        长度的非占位密钥，并关闭应用调试模式。回滚时可使用命令行镜像覆盖
        ``DOG_AGENT_API_IMAGE``，其余安全配置仍从私有环境文件读取。

    参数含义：
        values:
            从私有环境文件读取的配置映射。
        image_override:
            回滚等场景临时指定的镜像；为空时读取 ``DOG_AGENT_API_IMAGE``。

    返回值含义：
        list[str]:
            所有不合格原因；空列表表示预检通过。列表不会包含密钥原文。
    """

    errors: list[str] = []
    image = (image_override or values.get("DOG_AGENT_API_IMAGE", "")).strip()
    if not GHCR_VERSIONED_IMAGE_PATTERN.fullmatch(image):
        errors.append(
            "DOG_AGENT_API_IMAGE 必须是带三段固定版本号的 GHCR 镜像，"
            "例如 ghcr.io/owner/dog-agent-api:1.22.0；禁止 latest"
        )

    if not _is_true(values.get("API_AUTH_ENABLED")):
        errors.append("API_AUTH_ENABLED 必须设置为 true")

    api_key = values.get("API_AUTH_KEY", "").strip()
    if len(api_key) < 32 or api_key.lower() in PLACEHOLDER_API_KEYS:
        errors.append(
            "API_AUTH_KEY 必须使用至少 32 个字符的真实随机密钥，"
            "不能留空或使用占位值"
        )

    if not _is_true(values.get("API_RATE_LIMIT_ENABLED")):
        errors.append("API_RATE_LIMIT_ENABLED 必须设置为 true")

    if not _is_false(values.get("DEBUG")):
        errors.append("DEBUG 必须设置为 false")

    return errors


def _is_true(value: str | None) -> bool:
    """
    判断环境变量是否明确表示布尔真值。

    参数含义：
        value:
            待检查的可选环境变量字符串。

    返回值含义：
        bool:
            值忽略大小写后等于 ``true`` 时返回 True，否则返回 False。
    """

    return str(value or "").strip().lower() == "true"


def _is_false(value: str | None) -> bool:
    """
    判断环境变量是否明确表示布尔假值。

    参数含义：
        value:
            待检查的可选环境变量字符串。

    返回值含义：
        bool:
            值忽略大小写后等于 ``false`` 时返回 True，否则返回 False。
    """

    return str(value or "").strip().lower() == "false"


def _parse_args() -> argparse.Namespace:
    """
    解析生产配置预检命令行参数。

    参数含义：
        无。

    返回值含义：
        argparse.Namespace:
            包含环境文件路径和可选镜像覆盖值的命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="在拉取镜像和重建容器前校验 V1.22 生产部署配置。"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        required=True,
        help="需要检查的私有 .env 文件路径。",
    )
    parser.add_argument(
        "--image-override",
        default=None,
        help="回滚时临时使用的固定版本 GHCR 镜像。",
    )
    return parser.parse_args()


def main() -> None:
    """
    执行命令行生产配置预检。

    参数含义：
        无。

    返回值含义：
        None:
            配置合格时正常结束；不合格时仅输出字段级原因并以非零状态退出。
    """

    args = _parse_args()
    if not args.env_file.is_file():
        raise SystemExit(
            f"生产配置预检失败：环境文件不存在: {args.env_file}"
        )

    try:
        values = load_env_file(args.env_file)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"生产配置预检失败：{exc}") from exc

    errors = validate_release_environment(
        values,
        image_override=args.image_override,
    )
    if errors:
        print("生产配置预检失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(2)

    print("V1.22 生产部署配置预检通过；敏感配置值未输出。")


if __name__ == "__main__":
    main()
