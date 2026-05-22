from pydantic import BaseModel


# 参数校验
class WeatherArgs(BaseModel):

    city: str