import os
import glob
import time
import platform
import shutil
from uuid import uuid4
from loguru import logger
from app.utils import utils
from streamlit.runtime.uploaded_file_manager import UploadedFile
from app.models.file_info import LocalFileInfo
from typing import Union
import json

def open_task_folder(root_dir, task_id):
    """打开任务文件夹
    Args:
        root_dir: 项目根目录
        task_id: 任务ID
    """
    try:
        sys = platform.system()
        path = os.path.join(root_dir, "storage", "tasks", task_id)
        if os.path.exists(path):
            if sys == 'Windows':
                os.system(f"start {path}")
            if sys == 'Darwin':
                os.system(f"open {path}")
            if sys == 'Linux':
                os.system(f"xdg-open {path}")
    except Exception as e:
        logger.error(f"打开任务文件夹失败: {e}")

def cleanup_temp_files(temp_dir):
    """清理临时文件
    Args:
        temp_dir: 临时文件目录
    """
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                logger.debug(f"已清理临时文件: {file_path}")
            except Exception as e:
                logger.error(f"清理临时文件失败: {file_path}, 错误: {e}")

def get_file_list(directory, file_types=None, sort_by='ctime', reverse=True) -> list[LocalFileInfo]:
    """获取指定目录下的文件列表
    Args:
        directory: 目录路径
        file_types: 文件类型列表，如 ['.mp4', '.mov']
        sort_by: 排序方式，支持 'ctime'(创建时间), 'mtime'(修改时间), 'size'(文件大小), 'name'(文件名)
        reverse: 是否倒序排序
    Returns:
        list: 文件信息列表
    """
    if not os.path.exists(directory):
        return []
    
    files = []
    if file_types:
        for file_type in file_types:
            files.extend(glob.glob(os.path.join(directory, f"*{file_type}")))
    else:
        files = glob.glob(os.path.join(directory, "*"))
    
    file_list = []
    for file_path in files:
        try:
            file_stat = os.stat(file_path)
            file_name = os.path.basename(file_path)
            suffixs = file_name.split('.')
            file_info = LocalFileInfo(name=suffixs[0],suffix=(suffixs[-1] if len(suffixs) > 1 else ''),path=file_path,size=file_stat.st_size,ctime=file_stat.st_ctime,mtime=file_stat.st_mtime)
            file_list.append(file_info)
        except Exception as e:
            logger.error(f"获取文件信息失败: {file_path}, 错误: {e}")
    
    # 排序
    if sort_by in ['ctime', 'mtime', 'size', 'name']:
        file_list.sort(key=lambda x: getattr(x, sort_by), reverse=reverse)
    
    return file_list

def save_uploaded_file(uploaded_file:UploadedFile, save_dir, allowed_types=None):
    """保存上传的文件
    Args:
        uploaded_file: StreamlitUploadedFile对象
        save_dir: 保存目录
        allowed_types: 允许的文件类型列表，如 ['.mp4', '.mov']
    Returns:
        str: 保存后的文件路径，失败返回None
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        file_name, file_extension = os.path.splitext(uploaded_file.name)
        
        # 检查文件类型
        if allowed_types and file_extension.lower() not in allowed_types:
            logger.error(f"不支持的文件类型: {file_extension}")
            return None
        
        # 如果文件已存在，添加时间戳
        save_path = os.path.join(save_dir, uploaded_file.name)
        if os.path.exists(save_path):
            # delete
            os.remove(save_path)
            logger.info(f"remove already exists file: {save_path}")
            #timestamp = time.strftime("%Y%m%d%H%M%S")
            #new_file_name = f"{file_name}_{timestamp}{file_extension}"
            #save_path = os.path.join(save_dir, new_file_name)
        
        # 保存文件
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        
        logger.info(f"文件保存成功: {save_path}")
        return save_path
    
    except Exception as e:
        logger.error(f"保存上传文件失败: {e}")
        return None

def create_temp_file(prefix='tmp', suffix='', directory=None):
    """创建临时文件
    Args:
        prefix: 文件名前缀
        suffix: 文件扩展名
        directory: 临时文件目录，默认使用系统临时目录
    Returns:
        str: 临时文件路径
    """
    try:
        if directory is None:
            directory = utils.storage_dir("temp", create=True)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        temp_file = os.path.join(directory, f"{prefix}-{str(uuid4())}{suffix}")
        return temp_file
    
    except Exception as e:
        logger.error(f"创建临时文件失败: {e}")
        return None

def get_file_size(file_path, format='MB'):
    """获取文件大小
    Args:
        file_path: 文件路径
        format: 返回格式，支持 'B', 'KB', 'MB', 'GB'
    Returns:
        float: 文件大小
    """
    try:
        size_bytes = os.path.getsize(file_path)
        
        if format.upper() == 'B':
            return size_bytes
        elif format.upper() == 'KB':
            return size_bytes / 1024
        elif format.upper() == 'MB':
            return size_bytes / (1024 * 1024)
        elif format.upper() == 'GB':
            return size_bytes / (1024 * 1024 * 1024)
        else:
            return size_bytes
    
    except Exception as e:
        logger.error(f"获取文件大小失败: {file_path}, 错误: {e}")
        return 0

def ensure_directory(directory):
    """确保目录存在，如果不存在则创建
    Args:
        directory: 目录路径
    Returns:
        bool: 是否成功
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except Exception as e:
        logger.error(f"创建目录失败: {directory}, 错误: {e}")
        return False

def create_zip(files: list, zip_path: str, base_dir: str = None, folder_name: str = "demo") -> bool:
    """
    创建zip文件
    Args:
        files: 要打包的文件列表
        zip_path: zip文件保存路径
        base_dir: 基础目录，用于保持目录结构
        folder_name: zip解压后的文件夹名称，默认为frames
    Returns:
        bool: 是否成功
    """
    try:
        import zipfile
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                if not os.path.exists(file):
                    logger.warning(f"文件不存在，跳过: {file}")
                    continue
                    
                # 计算文件在zip中的路径，添加folder_name作为前缀目录
                if base_dir:
                    arcname = os.path.join(folder_name, os.path.relpath(file, base_dir))
                else:
                    arcname = os.path.join(folder_name, os.path.basename(file))
                
                try:
                    zipf.write(file, arcname)
                except Exception as e:
                    logger.error(f"添加文件到zip失败: {file}, 错误: {e}")
                    continue

        return True
        
    except Exception as e:
        logger.error(f"创建zip文件失败: {e}")
        return False

def save_data_to_file(data: Union[str, dict, list],file_name:str, file_path:str,file_type:str='json') -> bool:
    try:
        os.makedirs(file_path, exist_ok=True)
        full_path = os.path.join(file_path, f"{file_name}.{file_type}")
        
        if file_type.lower() == 'json':
            if isinstance(data, (dict, list)):
                json_data = json.dumps(data, ensure_ascii=False, indent=4)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(json_data)
            else:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(data)
        elif file_type.lower() == 'txt':
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
        
        return True
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f, strict=False)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在！")
    except json.JSONDecodeError:
        print("错误：文件内容不是有效的JSON格式！")
    except Exception as e:
        print(f"未知错误：{e}")
    return None

