from moviepy.editor import VideoFileClip
import os,time
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
import paddle
import atexit
from multiprocessing import Pool, cpu_count

video_path = "/Users/monkeygeek/Downloads/test.webm"
tmp_path = "/Users/monkeygeek/Downloads/tmp"
font_file_path = "/Users/monkeygeek/Documents/softProject/AIVideoTools/resource/fonts/STHeitiMedium.ttc"


# 全局变量用于保持进程内资源
process_data = {}

def init_process(video_path, font_file_path, use_gpu):
    """进程初始化函数，整个进程生命周期只执行一次"""
    # 初始化视频对象
    process_data['clip'] = VideoFileClip(video_path)
    
    # 初始化OCR模型
    process_data['ocr'] = PaddleOCR(
                    use_angle_cls=False, # 关闭方向检测,提升速度
                    det_model_dir="./resource/ocr_model/ch_PP-OCRv4_det_train",#区域检测
                    #rec_model_dir="./resource/ocr_model/ch_PP-OCRv4_rec_train",#方向识别
                    #cls_model_dir="./resource/ocr_model/ch_ppocr_mobile_v2.0_cls_train",#文本分类
                    rec=False,
                    cls=False,
                    use_gpu=use_gpu, # GPU开关
                    lang="ch" # 识别中文
                    )
    
    # 注册退出清理函数
    atexit.register(lambda: process_data['clip'].close())

def process_frame(args):
    """处理单个帧"""
    t, tmp_path = args
    try:
        # 从进程全局数据获取资源
        clip = process_data['clip']
        ocr = process_data['ocr']
        
        # 获取帧并保存
        frame = clip.get_frame(t)
        frame_path = os.path.join(tmp_path, f"frame_{t:.2f}s.png")
        Image.fromarray(frame).save(frame_path)
        
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
    except Exception as e:
        print(f"处理帧 {t} 时发生错误: {str(e)}")

def generate_tasks(video_path, tmp_path):
    """生成器函数，按需产生任务"""
    with VideoFileClip(video_path) as clip:
        for t, _ in clip.iter_frames(with_times=True):
            yield (t, tmp_path)

def test_video_subtitle_position_recognize(video_path: str, tmp_path: str, font_file_path: str):
    """优化后的多进程版本"""
    start_time = time.time()
    # https://paddlepaddle.github.io/PaddleOCR/latest/model/index.html
    
    # GPU配置
    use_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.get_device() == "gpu:0"
    
    # 创建进程池
    with Pool(
        processes=cpu_count(),
        initializer=init_process,
        initargs=(video_path, font_file_path, use_gpu)
    ) as pool:
        # 使用生成器按需产生任务
        task_generator = generate_tasks(video_path, tmp_path)
        
        # 使用imap实现流式处理
        for _ in pool.imap(process_frame, task_generator, chunksize=5):
            pass
    
    print(f"总执行时间：{time.time() - start_time:.6f} 秒")


if __name__ == '__main__':
   # test video subtitle position recognize
   test_video_subtitle_position_recognize(video_path,
                                          tmp_path,
                                          font_file_path)


