
import torch
import cv2, os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

from utils.metric_utils import _save_metrics, summarize_evaluation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--data-path", type=str, default="/home/ma-user/work/thomast/data/Kubric_new/test_final")
    parser.add_argument("--result-path", type=str, default="/home/ma-user/work/michaln/code/Deformable-3D-Gaussians/output/Kubric")
    parser.add_argument("--folder-name", type=str, default="test/ours_20000/renders")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--target-cam", type=str, default="cam2")
    args = parser.parse_args()


    for i in range(1, 101):
        scene_name = f"scene_{i:05d}"

        gt_file_path = f"{args.data_path}/{scene_name}/{args.target_cam}/rgba_00041.png"
        result_file_path = f"{args.result_path}/{scene_name}/{args.folder_name}/{args.target_cam}/rgba_00041.png"
        # print(f"reading files: {gt_file_path}, {result_file_path}")
        os.makedirs(f"{args.sample_dir}/{args.target_cam}/{scene_name}", exist_ok=True)
        gt_image = cv2.imread(gt_file_path)
        res_image = cv2.imread(result_file_path)
        # norm to 0-1
        gt_image = gt_image.astype('float32') / 255.0
        res_image = res_image.astype('float32') / 255.0
        _save_metrics(
            torch.from_numpy(gt_image).permute(2, 0, 1).unsqueeze(0).to(torch.float32).cuda(),
            torch.from_numpy(res_image).permute(2, 0, 1).unsqueeze(0).to(torch.float32).cuda(),
            [41],
            f"{args.sample_dir}/{args.target_cam}/{scene_name}",
            scene_name,
            custom_size=True,
            data_path=args.data_path,
            target_cam=args.target_cam
        )
        # break
    summarize_evaluation(f"{args.sample_dir}/{args.target_cam}")