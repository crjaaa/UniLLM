import asyncio

from typing import List
from dataclasses import dataclass

from .base import PublicModelService
from .types import HistoryItem


@dataclass
class ReplyMessage:
    model_id: str
    message_id: str
    index: int
    message: str
    stop: bool


def stream_responses(
        model_service: PublicModelService, model_id: str, prompt: str, history: List[HistoryItem], queue: asyncio.Queue
):
    """
    使用流式请求模型并将响应放入队列，为同时请求多个模型提供支持
    :param model_service: 使用的模型服务
    :param model_id: 使用的模型的ID，需要传入一个字符串，可自定义，主要用于在应用侧区分不同模型的响应
    :param prompt: 请求的提示词
    :param history: 请求的历史记录，注意部分模型可能不支持历史记录，这些模型会忽略该参数
    :param queue: 用于存放响应的队列
    :return:
    """
    if not model_service.streamable:
        reply_messages = model_service.run(prompt, history)
        asyncio.run(queue.put(ReplyMessage(model_id=model_id, index=0, message=reply_messages, stop=True)))
    else:
        index = 0
        reply_messages = ''
        for message in model_service.run_stream(prompt, history):
            asyncio.run(queue.put(ReplyMessage(model_id=model_id, index=index, message=message, stop=False)))
            reply_messages += message
            index += 1
        asyncio.run(queue.put(ReplyMessage(model_id=model_id, index=index, message=reply_messages, stop=True)))


async def test():
    _prompt = input("请输入请求的提示词：")
    # 测试同时请求多个模型
    from .public import OpenAIService, DashScopeService

    gpt3_5_service = OpenAIService({
        "api_key": "",
        "model": "gpt-3.5-turbo",
        "endpoint": "https://api.openai.com/v1"
    })
    gpt3_5_16k_service = OpenAIService({
        "api_key": "",
        "model": "gpt-3.5-turbo-16k",
        "endpoint": "https://api.openai.com/v1"
    })
    qianwen_service = DashScopeService({
        "model": "通义千问",
        "api_key": ""
    })

    _queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, stream_responses, gpt3_5_service, "gpt3.5", _prompt, [], _queue)
    loop.run_in_executor(None, stream_responses, gpt3_5_16k_service, "gpt3.5 16k", _prompt, [], _queue)
    loop.run_in_executor(None, stream_responses, qianwen_service, "dashscope", _prompt, [], _queue)

    model_count, stop_count = 3, 0
    while True:
        reply_message: ReplyMessage = await _queue.get()
        print(f'返回结果{reply_message}')
        if reply_message.stop:
            stop_count += 1
        if stop_count == model_count:
            print("所有模型都已返回结果")
            break


if __name__ == '__main__':
    asyncio.run(test())
