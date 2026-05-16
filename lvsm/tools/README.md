Tools
---

Some tools for different datasets or debugging on pose estimation, etc.


### Camera Traj

Reference from: [ReCamMaster](https://github.com/KwaiVGI/ReCamMaster/blob/main/vis_cam.py).

```bash
python plot_traj.py --normalize_scale --json_path /home/qingwen/workspace/LVSM/data/davis_nonorm/metadata/hike.json
```
demo:
![](../assets/docs/imgs/traj_demo.png)

### Plucker Ray Map

Visualize the Plucker ray map.

```bash
python tools/plot_plucker.py --scene_name 3aaed2e6422d7d57 --metadata_dir /home/qingwen/workspace/LVSM/data/realestate-10k/dataset/lvsm/test/metadata --evaluation_file /home/qingwen/workspace/LVSM/data/evaluation_index_re10k.json --rescale False
```
demo:
![](../assets/docs/imgs/plucker_demo.png)