from typing import Union

import orjson

from models import JsonData, JsonParser


class OrJsonParser(JsonParser):
    def from_json(self, json_data: Union[str, bytes]) -> JsonData:
        return orjson.loads(json_data)
