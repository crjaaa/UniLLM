# 放置于项目中
推荐使用 Git Submodule 的方式进行调用，这样可以保证模型服务的更新不会影响到项目的更新。

当然你也可以选择直接复制文件内容到项目中，但这样会导致模型服务更新不及时。

# Demo

## 1. 使用全部模型字典（推荐，因为接口都做了一致化处理）

```python
from service import public_services

service_name = "openai"

# 这样你就可以通过 service_name 获取到对应的模型服务了
config = {
  "api_key": "sk-wn0d8Rv************************Tos3UM",
  "model": "gpt-3.5-turbo",
  "endpoint": "https://api.openai.com/v1"
}
service = public_services.get(service_name).service_class  # 根据模型服务名称获取模型服务
config = service.verified_config(config)  # 验证模型服务配置
model_service = service(config)  # 根据模型服务配置初始化模型

# 模型服务调用：
# 模型服务可能支持流式调用，你可以通过 service.is_streaming 判断是否支持流式调用
# 非流式调用：
print(service.run("今天天气不错"))

# 流式调用：流式调用部分模型实现的效果可能是不同的，但在这里我们全部统一成了增量式输出
for result in service.run_stream("今天天气不错"):
    print(result)

```

## 2. 使用指定模型

```python
from service import OpenAIService

# 查看需要使用的参数配置：
OpenAIService.config_schemas()

# 参数配置验证：
config = {
  "api_key": "sk-wn0d8Rv************************Tos3UM",
  "model": "gpt-3.5-turbo",
  "endpoint": "https://api.openai.com/v1"
}
config = OpenAIService.verified_config(config)  # 此处会删除掉不需要的参数

# 模型服务初始化：
service = OpenAIService(config)

# 模型服务调用：
# 模型服务可能支持流式调用，你可以通过 service.is_streaming 判断是否支持流式调用
# 非流式调用：
print(service.run("今天天气不错"))

# 流式调用：流式调用部分模型实现的效果可能是不同的，但在这里我们全部统一成了增量式输出
for result in service.run_stream("今天天气不错"):
    print(result)

```
