from pydantic import BaseModel, Field


# 参数校验
class WeatherArgs(BaseModel):

    city: str = Field(
        min_length=1,
        description="要查询当前天气的城市名称，例如成都、北京或上海",
    )
