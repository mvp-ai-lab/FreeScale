import cv2
import numpy as np
import os

def create_static_video_from_image(image_path, output_path, duration_sec=3, fps=30):
    """
    从单张静态图片创建指定时长的视频。

    参数:
    image_path (str): 输入图片的路径。
    output_path (str): 输出视频的路径 (例如 'output.mp4')。
    duration_sec (int): 视频的时长（秒）。
    fps (int): 视频的帧率 (Frames Per Second)。
    """
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片文件未找到 at '{image_path}'")
        return

    # 1. 读取图片
    frame = cv2.imread(image_path)
    
    # 如果图片读取失败
    if frame is None:
        print(f"错误：无法读取图片 at '{image_path}'。请检查文件是否为有效的图片格式。")
        return

    # 2. 获取图片的尺寸
    height, width, layers = frame.shape
    size = (width, height)

    # 3. 定义视频编码器和创建 VideoWriter 对象
    # After (Recommended)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    if not out.isOpened():
        print(f"错误：无法创建视频文件 at '{output_path}'")
        return

    # 4. 计算总帧数并写入视频
    total_frames = duration_sec * fps
    print(f"正在创建视频...")
    print(f" - 时长: {duration_sec} 秒")
    print(f" - 帧率: {fps} FPS")
    print(f" - 总帧数: {total_frames}")

    for _ in range(total_frames):
        out.write(frame)

    # 5. 释放资源
    out.release()
    print(f"视频创建成功！已保存至: '{output_path}'")

# --- 使用示例 ---
if __name__ == "__main__":
    # --- 请修改以下参数 ---
    
    # 你的图片文件路径（请确保这张图片存在）
    input_image_file = '/home/qingwen/Pictures/kin.jpg' 
    
    # 输出的视频文件路径和名称
    output_video_file = f"/home/qingwen/Pictures/s2v_{input_image_file.split('.')[0].split('/')[-1]}.mp4"
    
    # 视频参数
    video_duration = 5  # 视频时长（秒）
    video_fps = 30      # 视频帧率

    # 运行函数来创建视频
    create_static_video_from_image(
        image_path=input_image_file,
        output_path=output_video_file,
        duration_sec=video_duration,
        fps=video_fps
    )