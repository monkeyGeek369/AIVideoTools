import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

def text_coordinate_recognize_video(video_frames_coordinates:dict, 
                                    video_width:int, 
                                    video_height:int, 
                                    fps:int,
                                    video_duration:float,
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
    if not all_coordinates:
        return None

    # detect text regions
    title_candidates, warning_candidates, subtitle_candidates = detect_text_regions(all_coordinates, video_width, video_height, warning_keywords)

    # detect title regions
    title_region = detect_fixed_text_regions(title_candidates, fps, video_duration, title_merge_distance)

    # detect warning regions
    warning_region = detect_fixed_text_regions(warning_candidates, fps, video_duration, warning_merge_distance)

    # detect subtitle regions
    ignore_bbos = []
    if title_region and title_region["bbox"]:
        ignore_bbos.append(title_region["bbox"])
    if warning_region and warning_region["bbox"]:
        ignore_bbos.append(warning_region["bbox"])
    subtitle_region = detect_high_frequency_regions(subtitle_candidates, ignore_bbos, fps, video_duration, subtitle_merge_distance)

    # build result
    return {
        "title": title_region,
        "warning": warning_region,
        "subtitle": subtitle_region
    }

def detect_text_regions(coordinates:list, video_width:int, video_height:int, warning_keywords:list[str]) -> tuple[dict,dict,list]:
    if not coordinates:
        return None,None,None

    # init result
    title_candidates = defaultdict(list)  # {text: [(frame_idx, bbox)]}
    warning_candidates = defaultdict(list)  # {text: [(frame_idx, bbox)]}
    subtitle_candidates = []  # [(frame_idx, bbox, text)]
    
    # thresholds
    center_threshold = video_width * 0.15  # 水平居中阈值
    top_region = video_height * 0.4  # 上半部分边界
    warning_side_threshold = video_width * 0.3  # 警示区域边界
    
    # detect
    for frame_idx, frame_data in coordinates:
        for (top_left, bottom_right, text) in frame_data:
            x1, y1 = top_left
            x2, y2 = bottom_right
            bbox = (x1, y1, x2, y2)
            center_x = (x1 + x2) / 2

            # title filter：upper + centered + fixed text
            if y2 <= top_region and abs(center_x - video_width/2) <= center_threshold:
                title_candidates[text].append((frame_idx, bbox))
            
            # warning filter：side + fixed text + contains warning keywords
            if any(kw in text for kw in warning_keywords):
                if center_x <= warning_side_threshold or center_x >= (video_width - warning_side_threshold):
                    warning_candidates[text].append((frame_idx, bbox))
            
            # subtitle filter：center + no fixed text
            if abs(center_x - video_width/2) <= center_threshold:
                subtitle_candidates.append((frame_idx, bbox, text))
    
    return title_candidates, warning_candidates, subtitle_candidates

def detect_fixed_text_regions(candidates:dict, fps:int, video_duration:float, title_merge_distance:int=100) -> dict:
    if not candidates:
        return None

    # filter: min frames required
    min_frames_required = (fps * int(video_duration)) / 2
    filtered_candidates = {text: frame_list for text, frame_list in candidates.items() if len(frame_list) >= min_frames_required }    
    if not filtered_candidates:
        return None
    
    # merge bboxes
    title_regions = []
    for text, frame_list in filtered_candidates.items():
        boxes = [bbox for _, bbox in frame_list]
        merged_bbox = merge_bboxes(boxes)
        title_regions.append({
            "sample_text": text,
            "bbox": merged_bbox,
            "stable_frames": len(set(index for index,_ in frame_list))
        })
    
    # merge title text
    merged_regions = []
    for region in title_regions:
        center = get_center(region["bbox"])
        matched = False
        for merged in merged_regions:
            merged_center = get_center(merged["bbox"])
            dist = ((center[0] - merged_center[0]) ** 2 + 
                    (center[1] - merged_center[1]) ** 2) ** 0.5
            if dist <= title_merge_distance:
                merged["bbox"] = merge_bboxes([merged["bbox"], region["bbox"]])
                merged["sample_text"] = f"{merged['sample_text']},{region['sample_text']}"
                merged["stable_frames"] += region["stable_frames"]
                matched = True
                break
        if not matched:
            merged_regions.append({
                "sample_text": region["sample_text"],
                "bbox": region["bbox"],
                "stable_frames": region["stable_frames"]
            })

    if not merged_regions:
        return None

    # get the best title region
    best_region = max(merged_regions, key=lambda x: x["stable_frames"])

    return {
        "sample_text": best_region["sample_text"],
        "bbox": best_region["bbox"],
        "stable_frames": best_region["stable_frames"]
    }

def detect_high_frequency_regions(datas:list[tuple[int,tuple[int,int,int,int],str]], ignore_bboxs:list[tuple[int,int,int,int]],fps:int, video_duration:float, merge_distance=100):
    # filter: ignore bboxs
    filtered_datas = ignore_bboxs_filter(datas, ignore_bboxs)
    if filtered_datas is None or len(filtered_datas) == 0:
        return None
    
    # dbscan: clusters of bboxes
    clusters = clusters_bboxes_dbscan(filtered_datas, merge_distance)
    if not clusters or len(clusters) == 0:
        return None
    
    # filter: ignore frames
    threshold = (fps * video_duration) / 2
    valid_clusters = [(len(cluster['stable_frames']),cluster) for cluster in clusters if len(cluster['stable_frames']) >= threshold]
    if not valid_clusters or len(valid_clusters) == 0:
        return None
    
    # select the best cluster
    best_cluster = max(valid_clusters, key=lambda x: x[0])[1]
    
    return {
        "bbox": list(best_cluster['bbox']),
        "stable_frames": len(best_cluster['stable_frames'])
    }

def merge_bboxes(bboxes):
    x1_min = min(bbox[0] for bbox in bboxes)
    y1_min = min(bbox[1] for bbox in bboxes)
    x2_max = max(bbox[2] for bbox in bboxes)
    y2_max = max(bbox[3] for bbox in bboxes)
    return [x1_min, y1_min, x2_max, y2_max]

def get_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

def ignore_bboxs_filter(datas:list[tuple[int,tuple[int,int,int,int],str]], ignore_bboxs:list[tuple[int,int,int,int]]) -> list:
    filtered_datas = []
    for frame_idx, bbox, text in datas:
        ignore = False
        for ignore_bbox in ignore_bboxs:
            if not is_intersection_small_enough(bbox, ignore_bbox):
                ignore = True
                break
        if not ignore:
            filtered_datas.append((frame_idx, bbox, text))

    return filtered_datas

def is_intersection_small_enough(rect1: tuple, rect2: tuple) -> bool:
    """
    判断两个矩形的相交面积是否 <= 两个矩形中任意一个面积的50%
    """
    if quick_reject(rect1, rect2):
        return True
    
    r1 = np.array(rect1)
    r2 = np.array(rect2)
    
    area1 = (r1[2]-r1[0])*(r1[3]-r1[1])
    area2 = (r2[2]-r2[0])*(r2[3]-r2[1])
    
    x_overlap = max(0, min(r1[2],r2[2]) - max(r1[0],r2[0]))
    y_overlap = max(0, min(r1[3],r2[3]) - max(r1[1],r2[1]))
    intersection = x_overlap * y_overlap
    
    return (intersection <= area1*0.5) and (intersection <= area2*0.5)

def quick_reject(bbox1, bbox2):
    return (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
            bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def clusters_bboxes_dbscan(filtered_datas, merge_distance):
    # get centers of bounding boxes
    centers = [ get_center(bbox) for frame_idx, bbox, _ in filtered_datas]
    
    # DBSCAN
    db = DBSCAN(eps=merge_distance, min_samples=3).fit(centers)
    labels = db.labels_
    
    # dbscan clusters
    clusters = {}
    for i, label in enumerate(labels):
        frame_idx, bbox, _ = filtered_datas[i]
        
        if label not in clusters:
            clusters[label] = {
                'bbox': list(bbox),
                'stable_frames': {frame_idx}
            }
        else:
            old_bbox = clusters[label]['bbox']
            clusters[label]['bbox'] = merge_bboxes([old_bbox, bbox])
            clusters[label]['stable_frames'].add(frame_idx)
    
    return list(clusters.values())

if __name__ == "__main__":
    """
    {
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
