import os
import argparse
import ffmpeg 
import imageio_ffmpeg

def make_video(rendered_image_dir, video_name, fps=10, gt_dir=None):
    ffmpeg_exe_path = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        if gt_dir and os.path.exists(gt_dir):
            stream_render = ffmpeg.input(os.path.join(rendered_image_dir, '%3d.png'), framerate=fps)
            stream_gt = ffmpeg.input(os.path.join(gt_dir, '%3d.png'), framerate=fps)
            
            stream = ffmpeg.filter([stream_render, stream_gt], 'hstack')
        else:
            stream = ffmpeg.input(os.path.join(rendered_image_dir, '%3d.png'), framerate=fps)

        (
            ffmpeg
            .filter(stream, 'pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2')
            .output(video_name, vcodec='libx264', pix_fmt='yuv420p')
            .run(cmd=ffmpeg_exe_path, overwrite_output=True, quiet=False)
        )
        print("[+] Success！")
    except ffmpeg.Error as e:
        print("[-] Failure：")
        print(e.stderr.decode('utf8'))

def main():
    parser = argparse.ArgumentParser(description="rendering images to video")
    parser.add_argument(
        "-d", "--dir", 
        dest="rendered_image_dir",
        type=str, 
        required=True, 
    )
    parser.add_argument(
        "-v", "--video", 
        dest="video_name",
        type=str, 
        required=True, 
        help="output path of video"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=5, 
    )
    parser.add_argument(
        "--gt", 
        dest="gt_dir",
        type=str, 
        default=None, 
        help="directory of ground truth images (optional)"
    )
    args = parser.parse_args()
    
    rendered_image_dir = args.rendered_image_dir
    video_name = args.video_name
    fps = args.fps

    make_video(rendered_image_dir, video_name, fps, args.gt_dir)

if __name__ == "__main__":
    main()