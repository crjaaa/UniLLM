from typing import Literal, Iterator, Type, List

from .types import (
    ConfigSchema,
    ConfigType,
    HistoryItem,
    ConfigSchemaType,
)
from ..errors import NotSupportedError, ConfigFieldMissingError, ConfigFieldTypeError, ConfigFieldInvalidError
from ..schemas import PublicServiceDetail


class BaseModelService:
    _config_schemas: List[ConfigSchema]
    name: str
    description: str
    type: Literal["private", "public"]
    config: ConfigType

    streamable: bool = False  # 是否支持流式调用

    def __init__(self, config: ConfigType):
        self.config = self.verified_config(config)

    @classmethod
    def config_schemas(cls) -> List[ConfigSchema]:
        """
        用于渲染配置表单的字段配置
        :return:
        """
        if cls._config_schemas is None:
            raise NotImplementedError("Model._config_schemas is not defined")
        return cls._config_schemas

    @classmethod
    def required_fields(cls) -> List[ConfigSchema]:
        """
        必填字段，用于判断输入配置是否完整
        :return:
        """
        return [schema for schema in cls.config_schemas() if schema.required]

    @classmethod
    def all_schemas(cls) -> List[ConfigSchema]:
        """
        所有字段，用于遍历读取并忽略未知字段
        :return:
        """
        return [schema for schema in cls.config_schemas()]

    @classmethod
    def verified_config(cls, config: ConfigType) -> ConfigType:
        """
        解析并校验用户输入配置，输出整体后的配置
        :param config:
        :return:
        """
        result = dict()
        for schema in cls._config_schemas:
            # 校验必选字段是否均存在
            if schema.required:
                if schema.when:
                    # 有条件必填
                    for when_field, when_value in schema.when.items():
                        if config.get(when_field) == when_value and schema.field not in config.keys():
                            raise ConfigFieldMissingError(
                                f"{schema.field} is required when {when_field} is {when_value}")
                else:
                    # 无条件必填
                    if schema.field not in config.keys():
                        raise ConfigFieldMissingError(f"{schema.field} is required but not provided")

            # 校验字段类型是否正确
            if schema.field in config.keys():
                if schema.type in [ConfigSchemaType.string, ConfigSchemaType.number, ConfigSchemaType.boolean]:
                    # 判断预期的类型是否正确
                    expected_type = {
                        ConfigSchemaType.string: str,
                        ConfigSchemaType.number: (int, float),
                        ConfigSchemaType.boolean: bool,
                    }[schema.type]
                    if not isinstance(config.get(schema.field), expected_type):
                        raise ConfigFieldTypeError(f"{schema.field} should be {schema.type}")
                elif schema.type == "enum" and schema.enum:
                    allowed_values = schema.enum.__members__.values()
                    if config.get(schema.field) not in allowed_values:
                        raise ConfigFieldInvalidError(f"{schema.field} should be in {allowed_values}")
                else:
                    raise NotImplementedError(f"ConfigSchemaType {schema.type} is not supported yet")
                result[schema.field] = config.get(schema.field)
        return result

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        """
        使用模型并返回对话结果
        """
        raise NotImplementedError("run() need to be implemented to use this model")

    # noinspection PyTypeChecker
    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        使用模型并返回对话结果
        """
        if not self.streamable:
            raise NotSupportedError("This model is not streamable")
        raise NotImplementedError("run_stream() need to be implemented to use this model")


class PublicModelService(BaseModelService):
    type = "public"


class PublicServiceInfo:
    """
    服务信息
    :argument key: 服务 ID
    :argument display_name: 服务显示名称
    :argument description: 服务描述
    :argument img: 服务图标
    :argument service_class: 服务类
    """
    def __init__(
            self, key: str, display_name: str, description: str, img: str, service_class: Type[PublicModelService]
    ):
        self.key = key
        self.display_name = display_name
        self.description = description
        self.img = img
        self.service_class = service_class

    def to_public_service(self):
        return PublicServiceDetail(
            key=self.key,
            display_name=self.display_name,
            description=self.description,
            img=self.img,
            config_schemas=[schema.to_dict() for schema in self.service_class.config_schemas()],
        )
