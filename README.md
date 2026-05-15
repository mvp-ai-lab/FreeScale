<br>
<p align="center">
<h1 align="center"><strong>🌲  The official implementation of "FreeScale: Scaling 3D Scenes via Certainty-Aware Free-View Generation" </strong></h1>
  <p align="center">
    <a href='https://jiangchenhan.github.io' target='_blank'>Chenhan JIANG</a>&emsp;
    <a href='' target='_blank'>Yu CHEN</a>&emsp;
    <a href='https://kin-zhang.github.io' target='_blank'>Qingwen ZHANG</a>&emsp;
    <a href='https://scholar.google.com/citations?user=9a1PjCIAAAAJ&hl=en' target='_blank'>Jifei SONG</a>&emsp;
    <a>Songcen Xu</a>&emsp;
    <a href='https://sites.google.com/view/dyyeung' target='_blank'>Dit-Yan YEUNG</a>&emsp;
    <a href='https://jiankangdeng.github.io' target='_blank'>Jiankang DENG</a>&emsp;
    <br>
    HKUST&emsp;KTH &emsp;University Of Surrey&emsp;Imperial College London
    <h2 align="center">CVPR 2026</h2>
  </p>
</p>


<div align="center">
<a href="https://arxiv.org/pdf/2604.10512"><img src="https://img.shields.io/badge/Paper-📖-blue?"></a> &nbsp;&nbsp;
<a href="https://mvp-ai-lab.github.io/FreeScale"><img src="https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white"></a>
</div>


## 💡 About
<!-- ![Teaser](assets/teaser.jpg) -->
<div style="text-align: justify;">
    <img src="assets/teaser-github.png" alt="Dialogue_Teaser" width=100% >


This is the **official repository** of **🌲 FreeScale**.  

The development of generalizable Novel View Synthesis (NVS) models is critically limited by the scarcity of large-scale training data featuring diverse and precise camera trajectories. While real-world captures are photorealistic, they are typically sparse and discrete. Conversely, synthetic data scales but suffers from a domain gap and often lacks realistic semantics. We introduce FreeScale, a novel framework that leverages the power of scene reconstruction to transform limited real-world image sequences into a scalable source of high-quality training data. Our key insight is that an imperfect reconstructed scene serves as a rich geometric proxy, but naively sampling from it amplifies artifacts. To this end, we propose a certainty-aware free-view sampling strategy identifying novel viewpoints that are both semantically meaningful and minimally affected by reconstruction errors. We demonstrate FreeScale's effectiveness by scaling up the training of feedforward NVS models, achieving a notable gain of 2.7 dB in PSNR on challenging out-of-distribution benchmarks. Furthermore, we show that the generated data can actively enhance per-scene 3D Gaussian Splatting optimization, leading to consistent improvements across multiple datasets. Our work provides a practical and powerful data generation engine to overcome a fundamental bottleneck in 3D vision.

</div>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#-getting-started">Getting started</a>
    </li>
    <li>
      <a href="#-fvsampling">Free-View Image Sampling</a>
    </li>
    <li>
      <a href="#-downstream_tasks">Enhance Downstream Tasks</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>


## 🚀 Getting Started
To set up **FreeScale**, follow the steps below:

### 1. Clone the repository and create conda environment

```
git clone https://github.com/mvp-ai-lab/FreeScale.git
cd FreeScale

conda create -n free_scale python=3.11
conda activate free_scale
```
### 2. Install PyTorch≥2.5.0 with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

### 3. Install requirements
```bash
pip install -r requirements.txt

pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
pip install git+https://github.com/AIBluefisher/gsplat.git --no-build-isolation
pip install git+https://github.com/fraunhoferhhi/PLAS.git


cd gaussian_splatting/
pip install submodules/fused-ssim --no-build-isolation
pip install submodules/FasterGSCudaBackend
pip install submodules/MortonEncoding --no-build-isolation
```

## 🎨 Free-View Image Sampling

Please replace the ```dataset.root_dir```, ```dataset.output_dir```, ```dataset.load_from``` and ```dataset.sampler_dir``` in ```gaussina_splatting/configs/``` with your path.
We use different init geometry for feedforward (custom_ff) and per-scene reconstruction (custom). For feedforward method, we use in-domain initial geometry ```dataset.val_interval=32```. For per-scene reconstruction, we use OOD setting, ```dataset.val_interval=0.3```.

### Commands
```bash
CONFIG_FILE = custom_ff # or custom
cd freescale/gaussian_splatting/

# Step1: reconstruct scene geomtry
INIT_PLY_TYPE = 'sparse' # or 'dense'
SCENE = [SCENE_PATH]
python train.py --config config/$CONFIG_FILE.yaml \
                --init_ply_type $INIT_PLY_TYPE \
                --scene $SCENE
                # [optional] --suffix 3dgs
# python denoise.py --config config/custom_fvg.yaml --start_index $START_ID 
python eval.py  --config config/$CONFIG_FILE.yaml \
                --init_ply_type $INIT_PLY_TYPE \
                --scene $SCENE --val 1 \
                # [optional] --suffix 3dgs

# Step2: FV Images Generation
python sample_trajs.py --config config/$CONFIG_FILE_fvg.yaml \
                       --init_ply_type $INIT_PLY_TYPE \
                       --scene $SCENE # [optional] --scene_list_file scene_list.txt
                      #  [optional] --suffix 3dgs

```

## 🎨 Downstream Tasks
### Enhance LVSM Training
First, transform saved camera.bin to camera.json. 
```bash
# Please replace "root" with your output_dir in configs/custom_ff_fvg.yaml
cd freescale/gaussian_splatting/
python prepare_camera_as_json.py
```
The evaluation index can be found in ```freescale/lvsm/data/```.
```bash
# For LVSM
# train
bash freescale/scripts/train_lvsm.sh

# evaluation
bash freescale/lvsm/test.sh
```

### Enhance Per-Scene Reconstruction
```bash
cd freescale/gaussian_splatting/
python train.py --config config/custom_fvg.yaml \
                --init_ply_type $INIT_PLY_TYPE \
                --scene $SCENE
                # [optional] --suffix 3dgs

# or use batch script
bash freescale/scripts/freeview_sampling.sh
```


### Data Format

The data should be organized in the following structure:

```
DATA_DIR/
├── {SCENE_ID}
│   ├── sparse
│   │   └── 0
│   │       ├── cameras.bin
│   │       ├── database.db
│   │       └── ...
│   ├── images
│   │   ├── [image_name]_000001.png
│   │   ├── [image_name]_000002.png
│   │   ├── ...
│   │   ├── [image_name]_000200.png
│   │   ├── [image_name]_000201.png
│   │   └── ...
│   └── depths
│   │   ├── [image_name]_000001.png
│   │   └── ...
```


# Acknowledgements
Our code is built on top of [DOGS](https://github.com/AIBluefisher/DOGS) and [RayZer](https://github.com/hwjiang1510/RayZer) codebases. 

# Citation
```
@inproceedings{jiang2026freescale,
  title={FreeScale: Scaling 3D Scenes via Certainty-Aware Free-View Generation},
  author={Jiang, Chenhan and Chen, yu and Zhang, Qingwen and Song, Jifei and Xu, Songcen and Yeung, Dit-Yan and Deng, Jiankang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2026}
}
```
