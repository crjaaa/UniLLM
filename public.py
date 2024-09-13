import base64
import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import List, Iterator
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time

import dashscope
import openai
import requests
from websockets.sync.client import connect
import sensenova
import zhipuai

from cache import redis_cache
from .base import PublicModelService
from .types import (
    ConfigSchema, ConfigSchemaType, HuggingFaceAPIType, OpenAIModelType, HistoryItem, WenXinModel, XingHuoModelVersion,
    DashScopeModel, SenseNovaModel, ZhiPuModel, QiHoo360Model
)


class OpenAIService(PublicModelService):
    name = "OpenAI 服务"
    description = "OpenAI 提供的服务，包括 GPT-3.5、GPT-4 等。"
    streamable = True

    _config_schemas = [
        ConfigSchema(
            field="api_key",
            type=ConfigSchemaType.string,
            description="OpenAI API Key",
            required=True,
        ),
        ConfigSchema(
            field="org_id",
            type=ConfigSchemaType.string,
            description="OpenAI 组织 ID",
            required=False,
        ),
        ConfigSchema(
            field="model",
            type=ConfigSchemaType.enum,
            description="OpenAI 模型",
            enum=OpenAIModelType,
            required=True,
        ),
        ConfigSchema(
            field="endpoint",
            type=ConfigSchemaType.string,
            description="OpenAI Endpoint",
            default="https://api.openai.com/v1",
            required=True,
        ),
    ]

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        openai.api_base = self.config["endpoint"]
        openai.api_key = self.config["api_key"]
        openai.organization = self.config.get("org_id", '')

        if history is None:
            history = []
        if parameters is None:
            parameters = {}

        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]

        completion = openai.ChatCompletion.create(
            model=self.config["model"], messages=messages, **parameters
        )
        return completion.choices[0].message.content  # type: ignore

    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        流式调用
        :param prompt:
        :param history:
        :param parameters:
        :return:
        """
        openai.api_base = self.config["endpoint"]
        openai.api_key = self.config["api_key"]
        openai.organization = self.config.get("org_id", '')

        if history is None:
            history = []
        if parameters is None:
            parameters = {}
        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]

        completion = openai.ChatCompletion.create(
            model=self.config["model"], messages=messages, stream=True, **parameters
        )
        for chunk in completion:
            for choice in chunk.choices:
                if choice.finish_reason == "stop":
                    return
                yield choice.delta.content  # type: str


class AzureOpenAIService(PublicModelService):
    name = "Azure OpenAI 服务"
    description = "Azure OpenAI 提供的服务。"
    streamable = True

    _config_schemas = [
        ConfigSchema(
            field="api_key",
            type=ConfigSchemaType.string,
            description="Azure OpenAI API Key",
            required=True,
        ),
        ConfigSchema(
            field="model",
            type=ConfigSchemaType.enum,
            description="Azure OpenAI 模型",
            enum=OpenAIModelType,
            required=True,
        ),
        ConfigSchema(
            field="endpoint",
            type=ConfigSchemaType.string,
            description="Azure OpenAI Endpoint",
            required=True,
        ),
        ConfigSchema(
            field="api_version",
            type=ConfigSchemaType.string,
            description="Azure OpenAI 版本",
            required=True,
        ),
    ]

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        openai.api_type = "azure"
        openai.api_version = self.config["api_version"]
        openai.api_base = self.config["endpoint"]
        openai.api_key = self.config["api_key"]

        if history is None:
            history = []
        if parameters is None:
            parameters = {}

        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]

        completion = openai.ChatCompletion.create(
            model=self.config["model"], messages=messages, **parameters
        )
        return completion.choices[0].message.content  # type: ignore

    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        流式调用
        :param prompt:
        :param history:
        :param parameters:
        :return:
        """
        openai.api_type = "azure"
        openai.api_version = self.config["api_version"]
        openai.api_base = self.config["endpoint"]
        openai.api_key = self.config["api_key"]

        if history is None:
            history = []
        if parameters is None:
            parameters = {}
        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]

        completion = openai.ChatCompletion.create(
            model=self.config["model"], messages=messages, stream=True, **parameters
        )
        for chunk in completion:
            for choice in chunk.choices:
                if choice.finish_reason == "stop":
                    return
                yield choice.delta.content  # type: str


class HuggingFaceService(PublicModelService):
    name = "Hugging Face 服务"
    description = "Hugging Face 提供的推理服务，可使用 Inference API 或 Inference Endpoints。"

    _config_schemas = [
        ConfigSchema(
            field="api_type",
            type=ConfigSchemaType.enum,
            description="Hugging Face API 类型",
            enum=HuggingFaceAPIType,
            required=True,
        ),
        ConfigSchema(
            field="endpoint",
            type=ConfigSchemaType.string,
            description="Hugging Face Inference Endpoint 地址",
            required=True,
            when={"api_type": HuggingFaceAPIType.endpoint},
        ),
        ConfigSchema(
            field="model",
            type=ConfigSchemaType.string,
            description="Hugging Face Hub 模型名称",
            required=True,
            when={"api_type": HuggingFaceAPIType.api},
        ),
        ConfigSchema(
            field="token",
            type=ConfigSchemaType.string,
            description="Hugging Face 用户 Token",
            required=True,
        ),
    ]

    @property
    def _endpoint(self) -> str:
        if self.config["api_type"] == HuggingFaceAPIType.api:
            return f"https://api-inference.huggingface.co/models/{self.config['model']}"
        elif self.config["api_type"] == HuggingFaceAPIType.endpoint:
            return self.config["endpoint"]
        else:
            raise Exception("Unreachable")

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        # convert list[HistoryItem] to two lists of texts
        past_user_inputs = []
        generated_responses = []
        if history:
            for item in history:
                if item.role == "user":
                    past_user_inputs.append(item.content)
                elif item.role == "assistant":
                    generated_responses.append(item.content)

        req = requests.post(
            self._endpoint,
            data={
                "inputs": {
                    "past_user_inputs": past_user_inputs,
                    "generated_responses": generated_responses,
                    "text": prompt,
                },
                "parameters": parameters,
            },
            headers={"Authorization": f"Bearer {self.config['token']}"},
        )
        if not req.ok:
            raise Exception("Request not OK")

        return req.json()["generated_text"]


class WenXinService(PublicModelService):
    name = "百度千帆大模型平台服务"
    description = "提供包括文心一言、BLOOMZ-7B、LLaMa 2 在内的多个模型服务。"
    streamable = True

    _access_token = None

    _config_schemas = [
        ConfigSchema(
            field="model",
            type=ConfigSchemaType.enum,
            description="服务模型",
            enum=WenXinModel,
            required=True,
        ),
        ConfigSchema(
            field="client_id",
            type=ConfigSchemaType.string,
            description="千帆平台创建应用的 client_id",
            required=True,
        ),
        ConfigSchema(
            field="client_secret",
            type=ConfigSchemaType.string,
            description="千帆平台创建应用的 client_secret",
            required=True,
        ),
    ]

    @property
    def access_token(self):
        """
        获取 access token
        :return:
        """
        if self._access_token is None:
            access_token = redis_cache.get(f'WENXIN/AccessToken/{self.config["client_id"]}')
            if access_token is None:
                req = requests.post(
                    'https://aip.baidubce.com/oauth/2.0/token',
                    params={
                        'grant_type': 'client_credentials',
                        'client_id': self.config["client_id"], 'client_secret': self.config['client_secret']
                    },
                    headers={'Content-Type': 'application/json', 'Accept': 'application/json'}
                )
                access_token = req.json()['access_token']
                redis_cache.set(f'WENXIN/AccessToken/{self.config["client_id"]}', access_token, ex=1296000)
            self._access_token = access_token
        return self._access_token

    @property
    def base_url(self) -> str:
        return 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/' + {
            'ERNIE-Bot': 'completions',
            'ERNIE-Bot-turbo': 'eb-instant',
            'BLOOMZ-7B': 'bloomz_7b1',
            'Qianfan-BLOOMZ-7B-compressed': 'qianfan_bloomz_7b_compressed',
            'Llama-2-7B': 'llama_2_7b',
            'Llama-2-13B': 'llama_2_13b',
            'Llama-2-70B': 'llama_2_70b',
            'Qianfan-Chinese-Llama-2-7B': 'qianfan_chinese_llama_2_7b',
            'ChatGLM2-6B-32K': 'chatglm2_6b_32k',
            'AquilaChat-7B': 'aquilachat_7b',
            'ERNIE-Bot-4': 'completions_pro',
        }[self.config['model']] + f'?access_token={self.access_token}'

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        if history is None:
            history = []
        if parameters is None:
            parameters = {}

        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]
        req = requests.post(self.base_url, json=dict(messages=messages, **parameters), stream=True)
        try:
            return req.json()['result']
        except KeyError:
            print(req.text)
            return '模型返回出现异常'

    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        流式调用
        :param prompt:
        :param history:
        :param parameters:
        :return:
        """
        if history is None:
            history = []
        if parameters is None:
            parameters = {}
        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]

        req = requests.post(self.base_url, json=dict(messages=messages, stream=True, **parameters), stream=True)
        for chunk in req.iter_lines():
            iter_text = chunk.decode('utf-8')
            if not iter_text.startswith('data:'):
                continue
            json_str = iter_text[6:]
            chunk_data = json.loads(json_str)
            yield chunk_data['result']
            if chunk_data['is_end']:
                return


class XingHuoService(PublicModelService):
    name = "讯飞星火认知大模型服务"
    description = "提供讯飞星火认知大模型 V1.5 和 V2.0 服务。"
    streamable = True

    _access_token = None

    _config_schemas = [
        ConfigSchema(
            field="version",
            type=ConfigSchemaType.enum,
            description="讯飞星火大模型版本",
            enum=XingHuoModelVersion,
            required=True,
        ),
        ConfigSchema(
            field="appid",
            type=ConfigSchemaType.string,
            description="讯飞开放平台服务应用 APPID",
            required=True,
        ),
        ConfigSchema(
            field="api_key",
            type=ConfigSchemaType.string,
            description="讯飞开放平台服务应用 APIKey",
            required=True,
        ),
        ConfigSchema(
            field="api_secret",
            type=ConfigSchemaType.string,
            description="讯飞开放平台服务应用 APISecret",
            required=True,
        ),
    ]

    @property
    def path(self):
        return urlparse(self.endpoint).path

    @property
    def endpoint(self):
        version = {'v1.5': 'v1.1', 'v2.0': 'v2.1'}[self.config['version']]
        return f"wss://spark-api.xf-yun.com/{version}/chat"

    @property
    def authorization_params(self):
        """
        获取 access token
        :return:
        """
        now = datetime.now()
        date = format_date_time(time.mktime(now.timetuple()))
        # date = format_datetime(datetime.now())
        signature_str = f'host: spark-api.xf-yun.com\ndate: {date}\nGET {self.path} HTTP/1.1'.encode()
        signature_sha = hmac.new(self.config['api_secret'].encode(), signature_str, digestmod=hashlib.sha256).digest()
        signature = base64.b64encode(signature_sha).decode()

        auth_str = f'api_key="{self.config["api_key"]}", ' \
                   f'algorithm="hmac-sha256", headers="host date request-line", ' \
                   f'signature="{signature}"'
        authorization = base64.b64encode(auth_str.encode()).decode()
        return urlencode({'authorization': authorization, 'date': date, 'host': 'spark-api.xf-yun.com'})

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        if history is None:
            history = []
        if parameters is None:
            parameters = {}
        return ''.join([content for content in self.run_stream(prompt, history, parameters)])

    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        流式调用
        :param prompt:
        :param history:
        :param parameters:
        :return:
        """
        if history is None:
            history = []
        if parameters is None:
            parameters = {}
        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]

        print(f'{self.endpoint}?{self.authorization_params}')
        with connect(f"{self.endpoint}?{self.authorization_params}") as ws:
            ws.send(json.dumps({
                "header": {'app_id': self.config['appid']},
                "parameter": {'chat': {'domain': 'general' if self.config['version'] == 'v1.5' else 'generalv2'}},
                "payload": {'message': {'text': messages}, **parameters},
            }))
            while True:
                resp = ws.recv()
                resp_json = json.loads(resp)
                if resp_json['header']['code'] != 0:  # 异常返回直接返回全部结果
                    print(resp)
                    yield resp_json['header']['message']
                    ws.close()
                    return

                for data in resp_json['payload']['choices']['text']:
                    yield data['content']
                if resp_json['header']['status'] == 2:
                    ws.close()
                    return


class DashScopeService(PublicModelService):
    name = "阿里云 DashScope 服务"
    description = "DashScope 提供包括通义千问、通义千问 7B、LLaMa 2、百川大模型在内的多个模型服务。"
    streamable = True

    _config_schemas = [
        ConfigSchema(
            field="model",
            type=ConfigSchemaType.enum,
            description="模型选择",
            enum=DashScopeModel,
            required=True,
            default=DashScopeModel.QWen,
        ),
        ConfigSchema(
            field="api_key",
            type=ConfigSchemaType.string,
            description="灵积模型服务 API Key",
            required=True,
        ),
    ]

    history_enable_model = [
        DashScopeModel.QWen, DashScopeModel.QWen_Plus, DashScopeModel.QWen_7B_Chat, DashScopeModel.SanLe,
        DashScopeModel.ChatGLM2_6B,
    ]

    @property
    def model_code(self) -> str:
        return {
            DashScopeModel.QWen: 'qwen-v1',
            DashScopeModel.QWen_Plus: 'qwen-plus-v1',
            DashScopeModel.QWen_7B: 'qwen-7b-v1',
            DashScopeModel.QWen_7B_Chat: 'qwen-7b-chat-v1',
            DashScopeModel.LLaMa2_7B: 'llama2-7b-v2',
            DashScopeModel.LLaMa2_13B: 'llama2-13b-v2',
            DashScopeModel.LLaMa2_7B_Chat: 'llama2-7b-chat-v2',
            DashScopeModel.LLaMa2_13B_Chat: 'llama2-13b-chat-v2',
            DashScopeModel.BaiChuan_7B: 'baichuan-7b-v1',
            DashScopeModel.ChatGLM2_6B: 'chatglm-6b-v2',
            DashScopeModel.SanLe: 'sanle-v1',
            DashScopeModel.ZiYa: 'ziya-llama-13b-v1',
            DashScopeModel.Dolly_LLaMa: 'dolly-12b-v2',
            DashScopeModel.BELLE_LLaMa: 'belle-llama-13b-2m-v1',
            DashScopeModel.MOSS: 'moss-moon-003-sft-v1',
            DashScopeModel.YuanYu: 'chatyuan-large-v2',
            DashScopeModel.BiLLa: 'billa-7b-sft-v1',
        }[self.config['model']]

    def format_history(self, history: List[HistoryItem] = None) -> List | None:
        if history is None:
            return None
        if self.config['model'] not in self.history_enable_model:
            return None
        # 通义千问平台的历史记录中 Role 与其他不同，需要单独处理
        history = [item for item in history if item.role != 'system']  # 移除 system 类型的历史记录
        print('History: ', history)
        history_rounds = []
        for index, message in enumerate(history):
            if message.role == 'user' and history[index + 1].role == 'assistant':
                if not message.content or not history[index + 1].content:
                    continue
                history_rounds.append({
                    'user': history[index].content,
                    'bot': history[index + 1].content,
                })

        if self.config['model'] == DashScopeModel.ChatGLM2_6B:
            # 当前历史记录两两分组，取出 content 填入列表
            return [message_round.values() for message_round in history_rounds]
        return history_rounds

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        dashscope.api_key = self.config["api_key"]

        if parameters is None:
            parameters = {}

        response = dashscope.Generation.call(
            model=self.model_code, prompt=prompt, history=self.format_history(history), **parameters
        )
        if response.status_code != 200:
            raise Exception(response.message)
        return response.output.text

    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        流式调用
        :param prompt:
        :param history:
        :param parameters:
        :return:
        """
        print(f"通义千问调用：{prompt}, {self.format_history(history)}")
        dashscope.api_key = self.config["api_key"]

        if parameters is None:
            parameters = {}

        responses = dashscope.Generation.call(
            model=self.model_code, prompt=prompt, history=self.format_history(history), stream=True, **parameters
        )
        before_str = ''  # 由于返回数据是覆盖而非增量，需要记录上一次的返回结果并删掉上一次的结果
        for response in responses:
            print(response.output)
            if response.output is None:
                print(response)
                continue
            if response.output.text.startswith(before_str):
                result_str = response.output.text[len(before_str):]
            else:
                result_str = response.output.text
            yield result_str
            before_str = response.output.text
            if response.output.finish_reason in ['stop', 'length']:
                return


class SenseNovaService(PublicModelService):
    name = "商汤日日新平台"
    description = "商汤日日新（SenseNova）官方大语言模型服务。"
    streamable = True

    _config_schemas = [
        ConfigSchema(
            field="model",
            type=ConfigSchemaType.enum,
            description="模型参数量",
            enum=SenseNovaModel,
            required=True,
            default=SenseNovaModel.xl,
        ),
        ConfigSchema(
            field="access_key_id",
            type=ConfigSchemaType.string,
            description="SenseNova 平台服务应用 Access Key ID",
            required=True,
        ),
        ConfigSchema(
            field="secret_access_key",
            type=ConfigSchemaType.string,
            description="SenseNova 平台服务应用 Secret Access Key",
            required=True,
        ),
    ]

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        sensenova.access_key_id = self.config["access_key_id"]
        sensenova.secret_access_key = self.config["secret_access_key"]

        if history is None:
            history = []
        if parameters is None:
            parameters = {}
        model_id = {
            SenseNovaModel.xl: 'nova-ptc-xl-v1',
            SenseNovaModel.xs: 'nova-ptc-xs-v1',
            SenseNovaModel.code: 'nova-ptc-l-v1-code'
        }[self.config['model']]
        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]

        completion = sensenova.ChatCompletion.create(model=model_id, messages=messages, **parameters)
        return completion.data.choices[0].message

    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        流式调用
        :param prompt:
        :param history:
        :param parameters:
        :return:
        """
        sensenova.access_key_id = self.config["access_key_id"]
        sensenova.secret_access_key = self.config["secret_access_key"]

        if history is None:
            history = []
        if parameters is None:
            parameters = {}
        model_id = {
            SenseNovaModel.xl: 'nova-ptc-xl-v1',
            SenseNovaModel.xs: 'nova-ptc-xs-v1',
            SenseNovaModel.code: 'nova-ptc-l-v1-code'
        }[self.config['model']]
        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]

        completion = sensenova.ChatCompletion.create(model=model_id, messages=messages, stream=True, **parameters)
        for chunk in completion:
            for choice in chunk.data.choices:
                if choice.delta:
                    yield choice.delta
                if choice.finish_reason == "stop":
                    return


class ZhiPuService(PublicModelService):
    name = "智谱 AI 开放平台"
    description = "ChatGLM 系列大语言模型服务。"
    streamable = True

    _config_schemas = [
        ConfigSchema(
            field="model",
            type=ConfigSchemaType.enum,
            description="模型选择",
            enum=ZhiPuModel,
            required=True,
            default=ZhiPuModel.ChatGLM_Std,
        ),
        ConfigSchema(
            field="api_key",
            type=ConfigSchemaType.string,
            description="智谱平台 API Key",
            required=True,
        ),
    ]

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        zhipuai.api_key = self.config["api_key"]

        if history is None:
            history = []
        if parameters is None:
            parameters = {'return_type': 'text'}
        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]

        response = zhipuai.model_api.invoke(
            model={
                ZhiPuModel.ChatGLM_Pro: 'chatglm_pro', ZhiPuModel.ChatGLM_Std: 'chatglm_std',
                ZhiPuModel.ChatGLM_Lite: 'chatglm_lite'
            }[self.config['model']], prompt=messages, **parameters
        )
        return response['data']['choices'][0]['content']

    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        流式调用
        :param prompt:
        :param history:
        :param parameters:
        :return:
        """
        zhipuai.api_key = self.config["api_key"]

        if history is None:
            history = []
        if parameters is None:
            parameters = {'return_type': 'text'}
        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]

        response = zhipuai.model_api.sse_invoke(
            model={
                ZhiPuModel.ChatGLM_Pro: 'chatglm_pro', ZhiPuModel.ChatGLM_Std: 'chatglm_std',
                ZhiPuModel.ChatGLM_Lite: 'chatglm_lite'
            }[self.config['model']], prompt=messages, **parameters
        )
        for event in response.events():
            if event.event == 'add':
                yield event.data
            elif event.event == "error" or event.event == "interrupted":
                raise Exception(event.data)
            elif event.event == "finish":
                yield event.data
                return
            else:
                print(event.event, event.data)
                yield event.data


class QiHoo360Service(PublicModelService):
    name = "360 智脑 API 开放平台服务"
    description = "提供360GPT_S2_V9模型。"
    streamable = True

    _config_schemas = [
        ConfigSchema(
            field="model",
            type=ConfigSchemaType.enum,
            description="服务模型",
            enum=QiHoo360Model,
            required=True,
            default=QiHoo360Model.GPT_S2_V9,
        ),
        ConfigSchema(
            field="api_key",
            type=ConfigSchemaType.string,
            description="360 智脑 API 平台 API Key",
            required=True,
        ),
    ]

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        if history is None:
            history = []
        if parameters is None:
            parameters = {}

        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]
        req = requests.post(
            'https://api.360.cn/v1/chat/completions', json=dict(
                model=self.config['model'], messages=messages, **parameters
            ), headers={
                'Authorization': f"Bearer {self.config['api_key']}",
                'Content-Type': 'application/json'
            })
        try:
            return req.json()['choices'][0]['message']['content']
        except KeyError:
            print(req.text)
            return '模型返回出现异常'

    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        流式调用
        :param prompt:
        :param history:
        :param parameters:
        :return:
        """
        if history is None:
            history = []
        if parameters is None:
            parameters = {}
        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]
        req = requests.post(
            'https://api.360.cn/v1/chat/completions', json=dict(
                model=self.config['model'], messages=messages, stream=True, **parameters
            ), headers={
                'Authorization': f"Bearer {self.config['api_key']}",
                'Content-Type': 'application/json'
            }, stream=True)
        for chunk in req.iter_lines():
            iter_text = chunk.decode('utf-8')
            if not iter_text.startswith('data:'):
                continue
            json_str = iter_text[6:]
            if json_str == '[DONE]':
                return
            chunk_data = json.loads(json_str)
            yield chunk_data['choices'][0]['delta']['content']
            if chunk_data['choices'][0]['finish_reason']:
                yield '模型返回出现异常，出现敏感词'
                return


class BaiChuanService(PublicModelService):
    name = "百川大模型服务"
    description = "百川2 53B 大模型服务"
    streamable = True

    _config_schemas = [
        ConfigSchema(
            field="api_key",
            type=ConfigSchemaType.string,
            description="百川大模型 API 平台 API Key",
            required=True,
        ),
        ConfigSchema(
            field="secret_key",
            type=ConfigSchemaType.string,
            description="百川大模型 API 平台 Secret Key",
            required=True,
        ),
    ]

    def headers(self, body_json: dict) -> dict:
        timestamp = str(int(time.time()))
        signature = hashlib.md5(f"{self.config['secret_key']}{json.dumps(body_json)}{timestamp}".encode()).hexdigest()

        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.config['api_key'],
            "X-BC-Request-Id": "your requestId",
            "X-BC-Timestamp": timestamp,
            "X-BC-Signature": signature,
            "X-BC-Sign-Algo": "MD5",
        }

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        if history is None:
            history = []
        if parameters is None:
            parameters = {}

        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]
        json_data = dict(model='Baichuan2-53B', messages=messages, parameters=parameters)
        req = requests.post('https://api.baichuan-ai.com/v1/chat', json=json_data, headers=self.headers(json_data))
        try:
            return req.json()['data']['messages'][0]['content']
        except KeyError:
            print(req.text)
            return '模型返回出现异常'

    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        流式调用
        :param prompt:
        :param history:
        :param parameters:
        :return:
        """
        if history is None:
            history = []
        if parameters is None:
            parameters = {}

        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]
        json_data = dict(model='Baichuan2-53B', messages=messages, parameters=parameters)
        req = requests.post(
            'https://api.baichuan-ai.com/v1/stream/chat', json=json_data, headers=self.headers(json_data), stream=True
        )
        for chunk in req.iter_lines():
            json_str = chunk.decode('utf-8')
            if not json_str:
                continue
            chunk_data = json.loads(json_str)
            yield chunk_data['data']['messages'][0]['content']
            if chunk_data['data']['messages'][0]['finish_reason'] == 'stop':
                return


class TencentHunYuanService(PublicModelService):
    name = "腾讯混元大模型服务"
    description = "腾讯混元大模型"
    streamable = True

    _config_schemas = [
        ConfigSchema(
            field="app_id",
            type=ConfigSchemaType.string,
            description="腾讯云账户 AppID",
            required=True,
        ),
        ConfigSchema(
            field="secret_id",
            type=ConfigSchemaType.string,
            description="腾讯云账户 SecretID",
            required=True,
        ),
        ConfigSchema(
            field='secret_key',
            type=ConfigSchemaType.string,
            description="腾讯云账户 SecretKey",
            required=True,
        ),
    ]

    def signature(self, data: dict) -> str:
        # 由于腾讯云对于格式化要求的比较特殊，对 messages 需要通过字符串的方式拼接，不要使用 json 库直接转字符串，json 库可能会对特殊字符进行转义
        messages = data.pop('messages')
        message_str = '[' + ','.join([
            f'{{"role":"{message["role"]}","content":"{message["content"]}"}}' for message in messages
        ]) + ']'
        params_dic = dict(messages=message_str, **data)
        # 将 params_dic 按照 key 字典序进行排序
        params_str = '&'.join([f'{key}={params_dic[key]}' for key in sorted(params_dic.keys())])
        sign_str = f'hunyuan.cloud.tencent.com/hyllm/v1/chat/completions?{params_str}'
        hmac_str = hmac.new(self.config['secret_key'].encode(), sign_str.encode(), digestmod=hashlib.sha1).digest()
        return base64.b64encode(hmac_str).decode()

    def run(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> str:
        if history is None:
            history = []
        if parameters is None:
            parameters = {}

        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]
        json_data = dict(
            app_id=self.config['app_id'], secret_id=self.config['secret_id'], messages=messages, stream=0,
            timestamp=int(time.time()), expired=int(time.time()) + 3600, **parameters,
        )
        req = requests.post('https://hunyuan.cloud.tencent.com/hyllm/v1/chat/completions', json=json_data, headers={
            'Authorization': self.signature(json_data),
            'Content-Type': 'application/json'
        })
        try:
            return req.json()['choices'][0]['messages']['content']
        except KeyError:
            print(req.text)
            return '模型返回出现异常'

    def run_stream(self, prompt: str, history: List[HistoryItem] = None, parameters: dict = None) -> Iterator[str]:
        """
        流式调用
        :param prompt:
        :param history:
        :param parameters:
        :return:
        """
        if history is None:
            history = []
        if parameters is None:
            parameters = {}

        messages = [{"role": item.role, "content": item.content} for item in history] + [
            {"role": "user", "content": prompt}
        ]
        json_data = dict(
            app_id=self.config['app_id'], secret_id=self.config['secret_id'], messages=messages, stream=1,
            timestamp=int(time.time()), expired=int(time.time()) + 3600, **parameters,
        )
        req = requests.post('https://hunyuan.cloud.tencent.com/hyllm/v1/chat/completions', json=json_data, headers={
            'Authorization': self.signature(json_data),
            'Content-Type': 'application/json'
        }, stream=True)
        for chunk in req.iter_lines():
            json_str = chunk.decode('utf-8')  # TODO: 由于腾讯云混元模型的授权还未获得，暂时无法测试，需要后续调试下
            if not json_str:
                continue
            chunk_data = json.loads(json_str)
            if 'error' in chunk_data:
                print(chunk_data['error']['message'])
            yield chunk_data['choices'][0]['delta']['content']
            if chunk_data['choices'][0]['finish_reason'] == 'stop':
                return
