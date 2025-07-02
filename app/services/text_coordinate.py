import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

def text_coordinate_recognize_video(video_frames_coordinates:dict, 
                                    video_width:int, 
                                    video_height:int, 
                                    warning_keywords:list[str], 
                                    title_merge_distance:int=100,
                                    warning_merge_distance:int=100,
                                    subtitle_merge_distance:int=100) -> dict:
    """Recognize text coordinates in video frames.
    video_frames_coordinates: dict the video frames coordinates
    {
    "0.0": {
        "index": 0,
        "coordinates": 
                [
                    ((11,35),(56,89),"text_content"),
                    ...
                ]
            }
    }
    video_width: int the width of the video
    video_height: int the height of the video
    warning_keywords: list of str the keywords to detect warning text
    title_merge_distance: int the distance to merge title text coordinates
    warning_merge_distance: int the distance to merge warning text coordinates
    subtitle_merge_distance: int the distance to merge subtitle text coordinates
    """
    # get all coordinates
    all_coordinates = [(frame_data.get("index"), frame_data.get("coordinates")) for t, frame_data in video_frames_coordinates.items()]


    pass

def detect_text_regions(coordinates:dict, video_width:int, video_height:int, warning_keywords:list[str]) -> tuple[dict,dict,list]:
    # init result
    title_candidates = defaultdict(list)  # {text: [(frame_idx, bbox)]}
    warning_candidates = defaultdict(list)  # {text: [(frame_idx, bbox)]}
    subtitle_candidates = []  # [(frame_idx, bbox, text)]
    
    # thresholds
    center_threshold = video_width * 0.15  # 水平居中阈值
    top_region = video_height * 0.4  # 上半部分边界
    bottom_region = video_height * 0.6  # 字幕区域边界
    warning_side_threshold = video_width * 0.3  # 警示区域边界
    
    # detect
    for frame_idx, frame_data in enumerate(coordinates):
        for (top_left, bottom_right, text) in frame_data:
            x1, y1 = top_left
            x2, y2 = bottom_right
            bbox = (x1, y1, x2, y2)
            center_x = (x1 + x2) / 2
            
            # 标题候选：上半部 + 水平居中 + 固定文本
            if y1 < top_region and abs(center_x - video_width/2) < center_threshold:
                title_candidates[text].append((frame_idx, bbox))
            
            # 警示候选：侧边位置 + 包含关键词
            if any(kw in text for kw in warning_keywords):
                if center_x < warning_side_threshold or center_x > (video_width - warning_side_threshold):
                    warning_candidates[text].append((frame_idx, bbox))
            
            # 字幕候选：底部区域 + 水平居中
            if y1 > bottom_region and abs(center_x - video_width/2) < center_threshold:
                subtitle_candidates.append((frame_idx, bbox, text))
    
    return title_candidates, warning_candidates, subtitle_candidates

def detect_media_regions(data, video_width, video_height, warning_keywords, merge_distance=100):
    
    # 结果存储
    result = {}
    
    # 检测标题区域
    if title_candidates:
        best_title = max(title_candidates.items(), 
                         key=lambda x: len(set(f[0] for f in x[1])))
        text, items = best_title
        frames = sorted({f[0] for f in items})
        stable_frames = max_continuous_frames(frames)
        
        # 计算合并后的边界框
        all_bboxes = [f[1] for f in items]
        merged_bbox = merge_bboxes(all_bboxes)
        
        result["title"] = {
            "bbox": merged_bbox,
            "stable_frames": stable_frames,
            "sample_text": text
        }
    
    # 检测警示区域
    if warning_candidates:
        best_warning = max(warning_candidates.items(), 
                          key=lambda x: len(set(f[0] for f in x[1])))
        text, items = best_warning
        frames = sorted({f[0] for f in items})
        stable_frames = max_continuous_frames(frames)
        
        # 计算合并后的边界框
        all_bboxes = [f[1] for f in items]
        merged_bbox = merge_bboxes(all_bboxes)
        
        result["warning"] = {
            "bbox": merged_bbox,
            "stable_frames": stable_frames
        }
    
    # 检测字幕区域
    if subtitle_candidates:
        # 聚类相似位置的字幕
        bbox_centers = []
        for idx, (frame_idx, bbox, _) in enumerate(subtitle_candidates):
            x1, y1, x2, y2 = bbox
            bbox_centers.append([(x1+x2)/2, (y1+y2)/2])
        
        # 使用DBSCAN聚类位置相近的文本块
        clustering = DBSCAN(eps=merge_distance, min_samples=3).fit(bbox_centers)
        clusters = defaultdict(list)
        
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # 忽略噪声点
                clusters[label].append(subtitle_candidates[idx])
        
        # 选择最大的稳定集群
        best_cluster = None
        max_stable_frames = 0
        for cluster in clusters.values():
            frames = sorted({item[0] for item in cluster})
            stable_frames = max_continuous_frames(frames)
            if stable_frames > max_stable_frames:
                max_stable_frames = stable_frames
                best_cluster = cluster
        
        if best_cluster:
            # 合并集群内所有边界框
            all_bboxes = [item[1] for item in best_cluster]
            merged_bbox = merge_bboxes(all_bboxes)
            
            result["subtitle"] = {
                "bbox": merged_bbox,
                "stable_frames": max_stable_frames
            }
    
    return result

def merge_bboxes(bboxes):
    x1_min = min(bbox[0] for bbox in bboxes)
    y1_min = min(bbox[1] for bbox in bboxes)
    x2_max = max(bbox[2] for bbox in bboxes)
    y2_max = max(bbox[3] for bbox in bboxes)
    return [x1_min, y1_min, x2_max, y2_max]

if __name__ == "__main__":
    """
    {
    "metadata": {
        "video_duration": 30.5,
        "resolution": [
            1920,
            1080
        ],
        "fps": 30
    },
    "frames": {
        "0": {
            "t": 0,
            "text_regions": [
                {
                    "text": "农场主花费高价买了只小奶牛",
                    "bbox": [
                        148,
                        209,
                        938,
                        283
                    ],
                    "type": "subtitle"
                },
                {
                    "text": "没想到长大后惊艳众人",
                    "bbox": [
                        232,
                        267,
                        847,
                        346
                    ],
                    "type": "subtitle"
                }
            ]
        }
    },
    "time_index": {
        "0.0": "0",
        "0.033": "1"
    },
    "fixed_regions": {
        "title": {
            "bbox": [
                100,
                200,
                950,
                300
            ],
            "stable_frames": 90,
            "sample_text": "农场主花费高价买了只小奶牛"
        },
        "warning": {
            "bbox": [
                50,
                50,
                500,
                100
            ],
            "stable_frames": 120
        },
        "subtitle": {
            "bbox": [
                200,
                800,
                1800,
                900
            ],
            "stable_frames": 300
        }
    }
}
    """
    pass
