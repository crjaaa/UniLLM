from typing import Dict

from .base import PublicServiceInfo, PublicModelService
from .public import AzureOpenAIService, HuggingFaceService, OpenAIService, WenXinService, XingHuoService, \
    DashScopeService, SenseNovaService, ZhiPuService, QiHoo360Service, BaiChuanService, TencentHunYuanService
from .types import (
    ConfigSchema,
    ConfigType,
    HistoryItem,
    ParameterType,
)

# when add a new public service, add it here
public_services: Dict[str, PublicServiceInfo] = {
    'azure-openai': PublicServiceInfo(
        key='azure-openai',
        display_name='Azure OpenAI',
        description='Azure OpenAI',
        img='azure-openai',
        service_class=AzureOpenAIService,
    ),
    'hugging-face': PublicServiceInfo(
        key='hugging-face',
        display_name='Hugging Face',
        description='Hugging Face',
        img='hugging-face',
        service_class=HuggingFaceService,
    ),
    'openai': PublicServiceInfo(
        key='openai',
        display_name='OpenAI',
        description='OpenAI',
        img='openai',
        service_class=OpenAIService,
    ),
    'wenxin': PublicServiceInfo(
        key='wenxin',
        display_name='百度千帆大模型平台',
        description='提供包括文心一言、BLOOMZ-7B 在内的多个模型服务。',
        img='wenxin',
        service_class=WenXinService,
    ),
    'xinghuo': PublicServiceInfo(
        key='xinghuo',
        display_name='讯飞星火认知大模型服务',
        description='提供讯飞星火认知大模型 V1.5 和 V2.0 服务。',
        img='xinghuo',
        service_class=XingHuoService,
    ),
    'dashscope': PublicServiceInfo(
        key='dashscope',
        display_name='阿里云灵积模型服务',
        description='提供通义千问、LLaMa2、ChatGLM2 等多个模型服务。',
        img='dashscope',
        service_class=DashScopeService,
    ),
    'sense': PublicServiceInfo(
        key='sense',
        display_name='商汤日日新平台模型服务',
        description='提供 SenseNova 两个模型服务。',
        img='sense',
        service_class=SenseNovaService,
    ),
    'zhipu': PublicServiceInfo(
        key='zhipu',
        display_name='智谱科技模型服务',
        description='提供智谱科技模型服务。',
        img='zhipu',
        service_class=ZhiPuService,
    ),
    '360': PublicServiceInfo(
        key='360',
        display_name='360 智脑 API 服务',
        description='提供 360 模型服务。',
        img='360',
        service_class=QiHoo360Service,
    ),
    'baichuan': PublicServiceInfo(
        key='baichuan',
        display_name='百川模型服务',
        description='提供百川2 53B 大模型服务。',
        img='baichuan',
        service_class=BaiChuanService,
    ),
    'tencent': PublicServiceInfo(
        key='tencent',
        display_name='腾讯混元大模型服务',
        description='提供腾讯混元大模型增强版服务',
        img='tencent',
        service_class=TencentHunYuanService,
    ),
}

__all__ = [
    'ConfigSchema', 'ConfigType', 'HistoryItem', 'ParameterType',
    'AzureOpenAIService', 'HuggingFaceService', 'OpenAIService', 'WenXinService', 'XingHuoService', 'DashScopeService',
    'public_services', 'PublicModelService'
]
