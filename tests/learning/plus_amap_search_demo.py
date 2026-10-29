"""高德地点搜索演示：支持城市关键词搜索和指定坐标周边搜索。"""

import getpass
import json
import os

import httpx


def search_places(*, api_key: str, city: str, keyword: str) -> dict:
    """调用高德地点关键字搜索，并返回接口的原始 JSON 数据。

    参数：
        api_key：高德 Web 服务类型密钥，用于认证本次请求。
        city：搜索城市，例如“成都市”。
        keyword：地点关键词，例如“宠物店”或具体商场名称。
    返回值：
        高德返回的字典，包含 status、info、infocode 和可能存在的 pois。
        HTTP 请求失败时抛出异常；业务是否成功由调用方检查 status。
    """
    params = {
        "key": api_key,
        "keywords": keyword,
        "region": city,
        "city_limit": "true",
        "page_size": 3,
        "page_num": 1,
    }

    with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
        response = client.get(
            "https://restapi.amap.com/v5/place/text",
            params=params,
        )
        response.raise_for_status()
        return response.json()


def search_nearby_places(
    *, api_key: str, location: str, keyword: str, radius: int = 3000
) -> dict:
    """查询指定坐标附近的地点，返回高德接口的原始 JSON 字典。

    参数：
        api_key：高德 Web 服务类型密钥。
        location：高德坐标字符串，格式为“经度,纬度”，使用英文逗号。
        keyword：地点关键词，例如“宠物店”。
        radius：搜索半径，单位米，默认 3000，允许范围为 0 至 50000。
    返回值：
        包含 status、info、infocode 和可能存在的 pois 的原始响应字典。
        HTTP 请求失败时抛出异常；本函数不查询路线或判断宠物入场政策。
    """
    params = {
        "key": api_key,
        "location": location,
        "keywords": keyword,
        "radius": radius,
        "sortrule": "distance",
        "page_size": 3,
        "page_num": 1,
    }

    with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
        response = client.get(
            "https://restapi.amap.com/v5/place/around",
            params=params,
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    api_key = os.environ.get("AMAP_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("请输入高德 Web 服务 Key（输入不显示）：").strip()
    if not api_key:
        raise SystemExit("未提供密钥，本次不发送请求。")

    mode = input("选择模式：1=城市搜索，2=周边搜索（默认2）：").strip() or "2"
    if mode not in {"1", "2"}:
        raise SystemExit("请选择1或2，本次不发送请求。")

    if mode == "1":
        city = input("搜索城市，直接回车使用成都市：").strip() or "成都市"
    else:
        location = input(
            "中心坐标，经度,纬度（默认104.109534,30.675249）："
        ).strip() or "104.109534,30.675249"
        print(f"以 {location} 为实验中心，搜索半径3000米；不是自动定位。")

    keyword = input("地点关键词，直接回车使用宠物店：").strip() or "宠物店"

    try:
        if mode == "1":
            result = search_places(api_key=api_key, city=city, keyword=keyword)
        else:
            result = search_nearby_places(
                api_key=api_key, location=location, keyword=keyword, radius=3000,
            )
    except httpx.HTTPError as exc:
        # 请求异常可能包含带密钥的 URL，因此只显示异常类型。
        raise SystemExit(f"地图请求失败：{type(exc).__name__}；请检查网络和接口状态。") from None
    except ValueError:
        raise SystemExit("地图响应无法解析为 JSON，请检查接口响应。") from None

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") != "1":
        print("接口业务检查失败，请查看上方 info 和 infocode 字段。")
    elif not result.get("pois"):
        print("接口调用成功，但本次没有找到地点。")
    else:
        print(f"接口调用成功，本页返回 {len(result['pois'])} 个地点。")
