import paddle
from paddleocr import PaddleOCR
import cv2
import threading
import queue
import atexit
import os
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
import shutil

# 全局变量用于保持进程内资源
process_data = {}
task_queue = queue.Queue(maxsize=100)

def init_paddleocr(use_gpu,max_batch_size):
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

def producer(video_path, tmp_path):
    """生成器函数，按需产生任务"""
    os.makedirs(tmp_path, exist_ok=True)    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_count / fps
        frame_path = os.path.join(tmp_path, f"frame_{t:.2f}s.png")
        cv2.imwrite(frame_path, frame)
        task_queue.put((t,frame_path))
        frame_count += 1
    cap.release()
    task_queue.put((None,None))

def consumer():
    """处理单个帧"""
    thread_local_results = {}
    try:
        # 从进程全局数据获取资源
        ocr = process_data['ocr']

        while True:
            t,frame_path = task_queue.get()
            coordinates = []
            if frame_path is None:  # 遇到结束标志
                task_queue.put((None,None))  # 通知其他消费者退出
                break
            
            # OCR检测
            result = ocr.ocr(frame_path, det=True, rec=False, cls=False)
            if not result or not result[0]:
                continue

            # 遍历结果并绘制矩形框
            for item in result[0]:
                top_left = tuple(map(int, item[0]))
                bottom_right = tuple(map(int, item[2]))

                # 检查坐标是否有效
                if top_left[0] < bottom_right[0] and top_left[1] < bottom_right[1]:
                    coordinates.append((top_left, bottom_right))
            
            thread_local_results[t] = coordinates
    except Exception as e:
        print(f"处理帧时发生错误: {str(e)}")
    return thread_local_results

def get_video_frames_coordinates(video_path:str,frame_tmp_path:str) -> dict:
    # GPU配置
    use_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.get_device() == "gpu:0"
    
    # 动态计算max_batch_size
    cap = cv2.VideoCapture(video_path)
    ret, sample_frame = cap.read()
    cap.release()
    if ret:
        h, w = sample_frame.shape[:2]
        frame_area = h * w
        max_batch_size = max(1, min(100, 4000 // (frame_area // 1000)))
    else:
        max_batch_size = 100

    # 主进程初始化OCR模型
    init_paddleocr(use_gpu,max_batch_size)
    
    # 启动生产者线程
    producer_thread = threading.Thread(target=producer, args=(video_path, frame_tmp_path))
    producer_thread.start()

    # 创建线程池（共享OCR模型）
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        # 获取异步执行结果：用于结果获取、状态获取、回调机制
        futures = [executor.submit(consumer) for _ in range(cpu_count())]
        frame_coordinates = {}
        for future in futures:
            thread_results = future.result()
            frame_coordinates.update(thread_results)
    
    # 等待生产者线程结束
    producer_thread.join()

    # 执行清理
    if 'ocr' in process_data:
        del process_data['ocr']
        
    # 删除临时帧文件
    shutil.rmtree(frame_tmp_path)

    return frame_coordinates


