from openai import OpenAI
import os
from loguru import logger
import json
# 设置 no_proxy 环境变量，使本地请求不经过代理
os.environ['no_proxy'] = '127.0.0.1,localhost'

def base_single_call_llm(base_url:str,api_key:str,model:str,prompt:str,content:str,temperature:float) -> str:
    result = None
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
        if result is None or len(result) == 0 or len(result.strip()) == 0:
            return None
    except Exception as e:
        result = None
        logger.warning(f"base single call llm error: {e}")
    finally:
        if client is not None:
            client.close()
            del client    
    return result

def chat_single_content(base_url:str,api_key:str,model:str,prompt:str,content:str,temperature:float,invalid_str:str,retry_count:int,is_remove_thinking:bool) -> str:
    while retry_count > 0:
        retry_count -= 1
        result = base_single_call_llm(base_url,api_key,model,prompt,content,temperature)
        if result is None:
            continue
        if invalid_str is not None and invalid_str in result:
            continue

        logger.info(f"chat single content result: {result}")
        # check think
        if "</think>" in result:
            ret_list = result.split("</think>")
            if len(ret_list) < 2:
                continue
            result = ret_list[1].strip()
        return result
    return None

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

def call_llm_get_list(base_url:str,api_key:str,model:str,prompt:str,content:str,retry_contents:list[str],temperature:float,retry_count:int) -> list[dict]:
    current_count = 0
    while current_count < retry_count:
        current_count += 1
        logger.info(f"call llm get list info: {current_count} execution now")
        result = base_single_call_llm(base_url,api_key,model,prompt,content,temperature)
        if result is None:
            continue
        if retry_contents is not None:
            for retry_content in retry_contents:
                if retry_content in result:
                    continue
        try:
            logger.info(f"call llm get list result: {result}")
            # check think
            if "</think>" in result:
                ret_list = result.split("</think>")
                if len(ret_list) < 2:
                    continue
                result = ret_list[len(ret_list)-1].strip()

            # result to list[dict]
            if result[-1] != ']':
                result = result + ']'
            list_dict = json.loads(result)
            if list_dict is None or len(list_dict) == 0:
                continue
            if not isinstance(list_dict, list):
                continue
            return list_dict
        except Exception as e:
            logger.warning(f"call llm get list error: {e}")
            continue
    return None

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


