from app.services import localhost_llm,subtitle
from app.utils import utils
import os,json


if __name__ == '__main__':
    subtitle_file_path = f"F:\download\村口最牛的情报中心,一不小心就身败名裂 #离谱 #搞笑 #万万没想到.srt"

    llm_result_list = subtitle.subtitle_llm_handler(base_url="http://localhost:1234/v1",
                                                api_key="lm-studio",
                                                model="qwen3-14b",
                                                prompt="你需要扮演一位严格遵守任务要求的字幕处理专家，更加具体的角色设定按照用户的要求进行。",
                                                title="村口最牛的情报中心,一不小心就身败名裂 #离谱 #搞笑 #万万没想到",
                                                subtitle_file_path=subtitle_file_path,
                                                temperature=0.7,
                                                retry_count=3)

    print(llm_result_list)
