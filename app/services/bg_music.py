from app.utils import utils
import os
from loguru import logger
import glob
import random

def get_bgm_type_list():
    bgm_path = utils.resource_dir("bgm")
    if not os.path.exists(bgm_path):
        os.makedirs(bgm_path)
        return []
    sub_dir_names = utils.get_subdir_names(bgm_path)
    if not sub_dir_names:
        return []
    return [(sub, os.path.join(bgm_path, sub)) for sub in sub_dir_names]

def get_all_bgm_file_paths(bgm_dir_path):
    if not os.path.exists(bgm_dir_path):
        return None
    mp3_files = glob.glob(os.path.join(bgm_dir_path, "*.mp3"))
    flac_files = glob.glob(os.path.join(bgm_dir_path, "*.flac"))
    return mp3_files + flac_files

def get_bgm_file(bgm_type, bgm_file,bgm_dir_path):
    """
    bgm_type: None | "random" | "custom"
    bgm_file: when bgm_type is "custom", this is the file path
    bgm_dir_path: when bgm_type is "random", this is the files dir path
    """

    # no bgm
    if bgm_type is None:
        return None

    # custom bgm
    if bgm_file == "custom":
        if os.path.exists(bgm_file) and os.path.isfile(bgm_file):
            if bgm_file.endswith(".mp3"):
                return bgm_file
            else:
                return None
        else:
            return None

    # random bgm
    if bgm_type == "random":
        if not os.path.exists(bgm_dir_path):
            return None
        bgm_paths = get_all_bgm_file_paths(bgm_dir_path)

        if not bgm_paths or len(bgm_paths) == 0:
            logger.warning(f"bgm not exist in dir {bgm_dir_path}")
            return None
        
        random.seed()
        return random.sample(bgm_paths, 1)[0]

    return None
