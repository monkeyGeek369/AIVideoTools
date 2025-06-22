import paddle
from paddleocr import PaddleOCR
import cv2
import threading
import queue
import os
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
import shutil
from app.services import mosaic,video
import streamlit as st
import sys
import multiprocessing
from app.utils import utils
import gc

# 在 macOS 上运行 PaddleOCR 多进程时必须：
if sys.platform == 'darwin':
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['USE_PPOCR_MULTI_PROCESS'] = '0'  # 禁用 PaddleOCR 内部多进程
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 全局变量
paddle_ocr = None
task_queue = queue.Queue(maxsize=100)

def init_paddleocr(use_gpu,max_batch_size):
    global paddle_ocr
    if sys.platform == 'darwin':
        rec_model_name = "PP-OCRv5_mobile_rec"
        rec_model_dir = os.path.join(utils.root_dir(), "resource", "ocr_model", "PP-OCRv5_mobile_rec_infer")
        det_model_name = "PP-OCRv5_mobile_det"
        det_model_dir = os.path.join(utils.root_dir(), "resource", "ocr_model", "PP-OCRv5_mobile_det_infer")
    else:
        rec_model_name = "PP-OCRv5_server_rec"
        rec_model_dir = os.path.join(utils.root_dir(), "resource", "ocr_model", "PP-OCRv5_server_rec_infer")
        det_model_name = "PP-OCRv5_server_det"
        det_model_dir = os.path.join(utils.root_dir(), "resource", "ocr_model", "PP-OCRv5_server_det_infer")

    paddle_ocr = PaddleOCR(
        device="gpu:0" if use_gpu else "cpu",
        use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
        use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
        use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
        text_recognition_model_name=rec_model_name, # 文本识别模型名称
        text_recognition_model_dir=rec_model_dir, # 文本识别模型路径
        text_detection_model_name=det_model_name, # 文本检测模型名称
        text_detection_model_dir=det_model_dir, # 文本检测模型路径
        precision='fp16',  # 显式指定精度
        enable_hpi=False, # 高性能推理是否启用(需要按照官网要求额外安装依赖)
        use_tensorrt=False, # 是否使用TensorRT加速(需要按照官网要求额外安装依赖)
        text_recognition_batch_size = max_batch_size # 文本识别批次大小
        )

def producer(video_path, tmp_path):
    """生成器函数，按需产生任务"""
    os.makedirs(tmp_path, exist_ok=True)    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # 当前帧位置
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_count / fps
        frame_path = os.path.join(tmp_path, f"frame_{frame_count}.png")
        cv2.imwrite(frame_path, frame)
        task_queue.put((frame_count,t,frame_path))
    cap.release()
    task_queue.put((None,None,None))

def consumer(batch_size):
    """处理单个帧"""
    thread_local_results = {}
    global paddle_ocr

    try:
        while True:
            # get batch size datas
            batch_size_datas = []
            while len(batch_size_datas) < batch_size:
                index,t,frame_path = task_queue.get()
                if frame_path is None:  # 遇到结束标志
                    task_queue.put((None,None,None))  # 通知其他消费者退出
                    break
                batch_size_datas.append((index,t,frame_path))

            # check batch size datas
            if len(batch_size_datas) <= 0:  # 遇到结束标志
                break
            
            # batch ocr process
            results = paddle_ocr.predict(input=[item[2] for item in batch_size_datas], 
                                        use_doc_orientation_classify=False,
                                        use_doc_unwarping=False,
                                        use_textline_orientation=False,
                                        text_rec_score_thresh=float(0.7))
            if results is None or len(results) <= 0:
                continue
            
            for batch_index,(index,t,frame_path) in enumerate(batch_size_datas):
                coordinates = []
                rec_scores = results[batch_index].get("rec_scores", [])
                rec_texts = results[batch_index].get("rec_texts", [])
                rec_boxes = results[batch_index].get("rec_boxes", [])

                # 遍历结果并绘制矩形框
                for score_item, text_item, box_item in zip(rec_scores, rec_texts, rec_boxes):
                    if score_item is None or score_item < 0.7:
                        continue
                    if text_item is None or len(text_item) < 2:
                        continue

                    box_positions_list = box_item.tolist()
                    positions = ((box_positions_list[0],box_positions_list[1]),
                                (box_positions_list[2],box_positions_list[3]))
                    text = text_item
                    top_left = tuple(map(int, positions[0]))
                    bottom_right = tuple(map(int, positions[1]))

                    # 检查坐标是否有效
                    if top_left[0] < bottom_right[0] and top_left[1] < bottom_right[1]:
                        coordinates.append((top_left, bottom_right, text))

                coord_result = {
                    "index":index,
                    "coordinates": coordinates
                }
                thread_local_results[t] = coord_result
    except Exception as e:
        print(f"处理帧时发生错误: {str(e)}")
    return thread_local_results

def get_video_frames_coordinates(video_path:str,frame_tmp_path:str) -> dict:
    global task_queue
    task_queue = queue.Queue(maxsize=100)
    
    # GPU配置
    use_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.get_device() == "gpu:0"
    
    # 主进程初始化OCR模型
    batch_size = 10
    init_paddleocr(use_gpu, batch_size)

    # 启动生产者线程
    producer_thread = threading.Thread(target=producer, args=(video_path, frame_tmp_path))
    producer_thread.start()

    # 消费者执行
    frame_coordinates = {}  
    consumer_result = consumer(batch_size)
    frame_coordinates.update(consumer_result)
    
    # 等待生产者线程结束
    producer_thread.join()

    # 删除临时帧文件
    shutil.rmtree(frame_tmp_path)

    # clean up
    safe_cleanup()
    task_queue = None

    return frame_coordinates

def get_frame_coordinates(frame_file_path:str,use_gpu:bool) -> list:
    init_paddleocr(use_gpu,100)

    coordinates = []
    result = paddle_ocr.ocr(frame_file_path, det=True, rec=False, cls=False)
    if not result or not result[0]:
        return coordinates

    for item in result[0]:
        top_left = tuple(map(int, item[0]))
        bottom_right = tuple(map(int, item[2]))

        if top_left[0] < bottom_right[0] and top_left[1] < bottom_right[1]:
            coordinates.append((top_left, bottom_right))
    return coordinates

def safe_cleanup():
    global paddle_ocr
    if paddle_ocr:
        try:
            # 尝试通过公开接口释放
            if hasattr(paddle_ocr, 'release'):
                paddle_ocr.release()
            
            # 强制删除内部组件（通过反射）
            for attr in ['_controller', '_det_model', '_rec_model', '_cls_model']:
                if hasattr(paddle_ocr, attr):
                    delattr(paddle_ocr, attr)
        except Exception as e:
            print(f"清理失败: {str(e)}")
        finally:
            paddle_ocr = None
            gc.collect()
    
    # 清理 GPU 缓存
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()
