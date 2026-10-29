"""独立 MCP 服务端教学：只查询内置教学卡片，不接入项目主图。

使用 MCP Python SDK 2.x；直接运行本文件启动本机 HTTP 服务。
卡片仅用于演示工具通信，不作为真实犬种知识或养护建议。
"""

from typing import Literal

from mcp.server import MCPServer
from pydantic import BaseModel


mcp = MCPServer("DogTeachingDemo")

TEACHING_CARDS = {
    "金毛": {"card_id": "teaching-golden", "topic": "金毛教学资料入口"},
    "柯基": {"card_id": "teaching-corgi", "topic": "柯基教学资料入口"},
}


class TeachingBreedCard(BaseModel):
    """功能：定义教学卡片的固定字段。

    参数：card_id 为卡片编号；topic 为教学资料主题。
    返回：经过字段校验的卡片对象。
    """

    card_id: str
    topic: str


class TeachingCardLookupResult(BaseModel):
    """功能：声明 MCP 工具的结构化查询结果。

    参数：status 表示找到或未找到；breed_name 为本次查询名称；
    card 为找到的卡片，未找到时为 None。
    返回：供 MCP 开发包生成输出结构、序列化返回数据的模型对象。
    """

    status: Literal["found", "not_found"]
    breed_name: str
    card: TeachingBreedCard | None


@mcp.tool()
def get_teaching_breed_card(breed_name: str) -> TeachingCardLookupResult:
    """功能：按犬种名称查询内置教学卡片，仅演示只读工具。

    参数：breed_name 为犬种名称，当前教学数据支持金毛、柯基。
    返回：包含查询状态、犬种名称和卡片的结构化模型；未找到时卡片为 None。
    """
    normalized_name = breed_name.strip()
    card = TEACHING_CARDS.get(normalized_name)
    return TeachingCardLookupResult(
        status="found" if card is not None else "not_found",
        breed_name=normalized_name,
        card=TeachingBreedCard(**card) if card is not None else None,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8765)
