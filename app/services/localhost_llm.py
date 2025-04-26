from openai import OpenAI
import os
from loguru import logger
# 设置 no_proxy 环境变量，使本地请求不经过代理
os.environ['no_proxy'] = '127.0.0.1,localhost'

def chat_single_content(base_url:str,api_key:str,model:str,prompt:str,content:str,temperature:float,invalid_str:str,retry_count:int) -> str:
    while retry_count > 0:
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content},
                ],
                temperature=temperature
            )

            result = completion.choices[0].message.content
            if invalid_str is not None and invalid_str in result:
                raise Exception(f"invalid_str: {invalid_str} in result: {result}")

            return result
        except Exception as e:
            logger.warning(f"chat_single_content error: {e}")
        finally:
            retry_count -= 1
            if client is not None:
                client.close()
                del client            

def check_llm_status(base_url:str,api_key:str,model:str) -> bool:
    client = None
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "hello"}
            ],
            temperature=0
        )

        return completion.choices[0].message.content is not None
    except Exception as e:
        return False
    finally:
        if client:
            client.close()
            del client


if __name__ == "__main__":
    content = """
1
00:00:00,980 --> 00:00:03,660
欢迎来到宝贝回家直播间
        
2
00:00:03,660 --> 00:00:05,780
姐 这是你的吗
        
3
00:00:05,780 --> 00:00:07,940
小宝 跟姐姐说谢谢
            """
    # message = chat_single_content(base_url="http://localhost:1234/v1",
    #                     api_key="lm-studio",
    #                     model="qwen2.5-14b-instruct",
    #                     prompt="你现在是一名中文视频字幕处理专家，给定中文字幕信息包含字幕index、字幕时间范围、字幕内容，当给到你字幕数据后希望你进行如下处理。-针对每一段字幕一定要重新生成字幕内容-新生成的字幕内容要与原字幕上下文语意相同但文字要有差异-新生成的字幕要满足原视频时间范围-直接输出处理后的中文字幕结果，无需输出其它内容-输出格式要与原格式相同-禁止带标点符号，可以用空格代替",
    #                     content=content,
    #                     temperature=0.7)
    # print(message)

    result = check_llm_status(base_url="http://localhost:1234/v1",
                    api_key="lm-studio",
                    model="qwen2.5-14b-instruct")
    
    print(result)


