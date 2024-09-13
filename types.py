from collections.abc import Mapping
from enum import EnumType, StrEnum
from typing import Any, Literal, NamedTuple, Optional, TypeAlias

# Note: "Any" to escape from static type checking. Type check of configs should be strictly done in runtime.
KeyType: TypeAlias = Any
ConfigType: TypeAlias = Mapping[str, KeyType]
ParameterType: TypeAlias = Mapping[str, KeyType]


class ConfigSchemaType(StrEnum):
    string = "string"
    number = "number"
    boolean = "boolean"
    enum = "enum"


class ConfigSchema(NamedTuple):
    """
    用于描述模型的配置
    """

    field: str
    type: ConfigSchemaType
    description: str
    required: bool
    # "required" has effect only when "when" is satisfied
    when: Optional[ConfigType] = None
    enum: Optional[EnumType] = None
    default: Optional[KeyType] = None

    def to_dict(self) -> dict[str, KeyType]:
        return {
            "field": self.field,
            "type": self.type.value,
            "description": self.description,
            "required": self.required,
            "when": self.when,
            "enum": [e.value for e in self.enum] if self.enum is not None else None,
            "default": self.default,
        }


class OpenAIModelType(StrEnum):
    gpt_4 = "gpt-4"
    gpt_4_32k = "gpt-4-32k"
    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_3_5_turbo_16k = "gpt-3.5-turbo-16k"
    gpt_3_5_turbo_0301 = "gpt-3.5-turbo-0301"
    gpt_3_5_turbo_0613 = "gpt-3.5-turbo-0613"
    gpt_3_5_turbo_16k_0613 = "gpt-3.5-turbo-16k-0613"


class HuggingFaceAPIType(StrEnum):
    api = "api"
    endpoint = "endpoint"


class HistoryItem(NamedTuple):
    """
    对话历史记录
    """
    role: Literal["user", "assistant", "system"]
    content: str


class WenXinModel(StrEnum):
    ERNIE_Bot = "ERNIE-Bot"
    ERNIE_Bot_turbo = "ERNIE-Bot-turbo"
    BLOOMZ_7B = "BLOOMZ-7B"
    QIANFAN_BLOOMZ_7B_COMPRESSED = 'Qianfan-BLOOMZ-7B-compressed'
    LLAMA2_7B = "Llama-2-7B"
    LLAMA2_13B = "Llama-2-13B"
    LLAMA2_70B = "Llama-2-70B"

    QIANFAN_CN_LLAMA2_7B = 'Qianfan-Chinese-Llama-2-7B'
    CHATGLM2_6B_32K = 'ChatGLM2-6B-32K'
    AQUILACHAT_7B = 'AquilaChat-7B',
    ERNIE_Bot_4 = "ERNIE-Bot-4"


class XingHuoModelVersion(StrEnum):
    V1_5 = "v1.5"
    V2_0 = "v2.0"


class DashScopeModel(StrEnum):
    QWen = "通义千问"
    QWen_Plus = "通义千问增强版"
    QWen_7B = "通义千问 7B"
    QWen_7B_Chat = "通义千问 7B Chat"
    LLaMa2_7B = "LLaMa2 7B"
    LLaMa2_13B = "LLaMa2 13B"
    LLaMa2_7B_Chat = "LLaMa2 7B Chat"
    LLaMa2_13B_Chat = "LLaMa2 13B Chat"
    BaiChuan_7B = "百川 7B"
    ChatGLM2_6B = "ChatGLM2 6B"
    SanLe = "智海三乐教育大模型"
    ZiYa = "姜子牙通用大模型"
    Dolly_LLaMa = "Dolly 12B"
    BELLE_LLaMa = "BELLE 13"
    MOSS = "MOSS 大模型"
    YuanYu = "元语大模型"
    BiLLa = "BiLLa 大模型"


class SenseNovaModel(StrEnum):
    xl = "大参数量版"
    xs = "小参数量版"
    code = '大参数量代码生成版'


class ZhiPuModel(StrEnum):
    ChatGLM_Pro = "chatglm_pro"
    ChatGLM_Std = "chatglm_std"
    ChatGLM_Lite = "chatglm_lite"


class QiHoo360Model(StrEnum):
    GPT_S2_V9 = '360GPT_S2_V9'
