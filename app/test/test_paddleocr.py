from moviepy.editor import VideoFileClip
import os,time
import threading
import queue
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())  # 检查 CUDA 是否可用
# print(torch.backends.cudnn.is_available())  # 检查 cuDNN 是否可用
# print(torch.backends.cudnn.version())  # 检查 cuDNN 是否可用
# print(torch.backends.cudnn.enabled)  # 检查 cuDNN 是否被启用
from PIL import Image, ImageDraw, ImageFont
import paddle
#paddle.utils.run_check()
from paddleocr import PaddleOCR
import atexit
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool

# mac
# video_path = "/Users/monkeygeek/Downloads/test.webm"
# tmp_path = "/Users/monkeygeek/Downloads/tmp"
# font_file_path = "/Users/monkeygeek/Documents/softProject/AIVideoTools/resource/fonts/STHeitiMedium.ttc"

# windows
video_path = "F:\download\\test.webm"
tmp_path = "F:\download\\tmp"
font_file_path = "F:\download\STHeitiMedium.ttc"

# 全局变量用于保持进程内资源
process_data = {}
task_queue = queue.Queue(maxsize=100)
result_queue = queue.Queue()

def init_process(font_file_path, use_gpu,max_batch_size):
    """进程初始化函数，整个进程生命周期只执行一次"""

    # 初始化OCR模型
    process_data['ocr'] = PaddleOCR(
                    use_angle_cls=False, # 关闭方向检测,提升速度
                    det_model_dir="./resource/ocr_model/ch_PP-OCRv4_det_train",#区域检测
                    #rec_model_dir="./resource/ocr_model/ch_PP-OCRv4_rec_train",#方向识别
                    #cls_model_dir="./resource/ocr_model/ch_ppocr_mobile_v2.0_cls_train",#文本分类
                    rec=False,
                    cls=False,
                    layout=False,  # 关闭布局分析（不需要结构）
                    table=False,   # 关闭表格识别（不需要表格）
                    precision='int8',  # 显式指定精度
                    use_gpu=use_gpu, # GPU开关
                    max_batch_size=max_batch_size, # 最大批次（将多个图像合并成一个批次，此参数设定了批次的最大容量）
                    lang="ch" # 识别中文
                    )
    
    # 注册退出清理函数
    def cleanup():
        if 'ocr' in process_data:
            del process_data['ocr']  # 显式释放OCR模型
    atexit.register(cleanup)

def consumer():
    """处理单个帧"""
    try:
        # 从进程全局数据获取资源
        ocr = process_data['ocr']

        while True:
            frame_path = task_queue.get()
            if frame_path is None:  # 遇到结束标志
                task_queue.put(None)  # 通知其他消费者退出
                break
            
            # OCR检测
            result = ocr.ocr(frame_path, det=True, rec=False, cls=False)
            if not result or not result[0]:
                return

            # 绘制检测框
            image = Image.open(frame_path)
            draw = ImageDraw.Draw(image)
            for item in result[0]:
                top_left = tuple(map(int, item[0]))
                bottom_right = tuple(map(int, item[2]))
                
                if top_left[0] < bottom_right[0] and top_left[1] < bottom_right[1]:
                    draw.rectangle([top_left, bottom_right], outline="red", width=6)
            
            image.save(frame_path)
            image.close()
            result_queue.put(1)  # 标记任务完成
    except Exception as e:
        print(f"处理帧时发生错误: {str(e)}")

def producer(video_path, tmp_path):
    """生成器函数，按需产生任务"""
    os.makedirs(tmp_path, exist_ok=True)
    with VideoFileClip(video_path) as clip:
        for t, frame in clip.iter_frames(with_times=True, dtype='uint8'):
            frame_path = os.path.join(tmp_path, f"frame_{t:.2f}s.png")
            Image.fromarray(frame).save(frame_path)
            task_queue.put(frame_path)  # 将任务放入队列
    task_queue.put(None)  # 结束标志

def test_video_subtitle_position_recognize(video_path: str, tmp_path: str, font_file_path: str):
    """优化后的多进程版本"""
    start_time = time.time()
    # https://paddlepaddle.github.io/PaddleOCR/latest/model/index.html
    
    # GPU配置
    use_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.get_device() == "gpu:0"
    
    # 在 init_process 中动态设置
    # sample_frame = next(generate_tasks(video_path, tmp_path))  # 获取第一帧示例
    # frame_size = Image.open(sample_frame).size
    # max_batch_size = max(1, min(100, 4000 // (frame_size[0] * frame_size[1] // 1000)))  # 经验公式

    # 主进程初始化OCR模型
    init_process(font_file_path, use_gpu,100)

    # 启动生产者线程
    producer_thread = threading.Thread(target=producer, args=(video_path, tmp_path))
    producer_thread.start()

    # 创建线程池（共享OCR模型）
    with ThreadPool(processes=cpu_count()) as pool:
        pool.map(lambda x: consumer(), range(cpu_count()))  # 每个线程执行consumer
    
    producer_thread.join()
    # 执行清理
    if 'ocr' in process_data:
        del process_data['ocr']
    print(f"总执行时间：{time.time() - start_time:.6f} 秒")


if __name__ == '__main__':
   # test video subtitle position recognize
   test_video_subtitle_position_recognize(video_path,
                                          tmp_path,
                                          font_file_path)


