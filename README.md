<br/>
<div align="center">
<h1 align="center"><strong>🌲 FreeScale</strong></h1>
<h3 align="center">Scaling 3D Scenes via Certainty-Aware Free-View Generation</h3>
  <p align="center">
    <strong>CVPR 2026</strong><br/>
    <a href='https://jiangchenhan.github.io' target='_blank'>Chenhan JIANG</a>&emsp;
    <a href='https://aibluefisher.github.io' target='_blank'>Yu CHEN</a>&emsp;
    <a href='https://kin-zhang.github.io' target='_blank'>Qingwen ZHANG</a>&emsp;
    <a href='https://scholar.google.com/citations?user=9a1PjCIAAAAJ&hl=en' target='_blank'>Jifei SONG</a>&emsp;
    <a>Songcen Xu</a>&emsp;
    <a href='https://sites.google.com/view/dyyeung' target='_blank'>Dit-Yan YEUNG</a>&emsp;
    <a href='https://jiankangdeng.github.io' target='_blank'>Jiankang DENG</a>
    <br/>
    <small>HKUST • NUS • KTH • University of Surrey • Imperial College London</small>
  </p>
</div>

<div align="center">
  <a href="https://arxiv.org/pdf/2604.10512"><img src="https://img.shields.io/badge/Paper-📖-blue.svg?style=for-the-badge&logo=arxiv&logoColor=white"></a>&nbsp;
  <a href="https://mvp-ai-lab.github.io/FreeScale"><img src="https://img.shields.io/badge/Project-Page-blue.svg?style=for-the-badge&logo=googlechrome&logoColor=white"></a>&nbsp;
  <a href="https://github.com/mvp-ai-lab/FreeScale"><img src="https://img.shields.io/badge/Code-GitHub-black.svg?style=for-the-badge&logo=github&logoColor=white"></a>
</div>

<br/>

## 📖 Abstract

<div align="justify">
  <img src="assets/teaser-github.png" alt="FreeScale Teaser" width="100%"/>
  <br/><br/>
  The development of generalizable Novel View Synthesis (NVS) models is critically limited by the scarcity of large-scale training data featuring diverse and precise camera trajectories. While real-world captures are photorealistic, they are typically sparse and discrete. Conversely, synthetic data scales but suffers from a domain gap and often lacks realistic semantics. 
  <br/><br/>
  We introduce <strong>FreeScale</strong>, a novel framework that leverages the power of scene reconstruction to transform limited real-world image sequences into a scalable source of high-quality training data. Our key insight is that an imperfect reconstructed scene serves as a rich geometric proxy, but naively sampling from it amplifies artifacts. To this end, we propose a <strong>certainty-aware free-view sampling</strong> strategy identifying novel viewpoints that are both semantically meaningful and minimally affected by reconstruction errors. 
  <br/><br/>
  We demonstrate FreeScale's effectiveness by scaling up the training of feedforward NVS models, achieving a notable gain of <strong>2.7 dB in PSNR</strong> on challenging out-of-distribution benchmarks. Furthermore, we show that the generated data can actively enhance per-scene 3D Gaussian Splatting optimization, leading to consistent improvements across multiple datasets. Our work provides a practical and powerful data generation engine to overcome a fundamental bottleneck in 3D vision.
</div>

<br/>

## 📑 Table of Contents

<details open>
<summary><strong>Click to expand</strong></summary>

1. [🚀 Getting Started](#-getting-started)
2. [🎨 Free-View Image Sampling](#-free-view-image-sampling)
3. [🔄 Enhance Downstream Tasks](#-enhance-downstream-tasks)
   - [Enhance LVSM Training](#enhance-lvsm-training)
   - [Enhance Per-Scene Reconstruction](#enhance-per-scene-reconstruction)
4. [📁 Data Format](#-data-format)
5. [🙏 Acknowledgements](#-acknowledgements)
6. [📝 Citation](#-citation)

</details>

<br/>

## 🚀 Getting Started

### Prerequisites

- **Python** ≥ 3.11
- **PyTorch** ≥ 2.5.0 with CUDA support
- **CUDA** ≥ 12.8

### Installation

```bash
# Clone the repository
git clone https://github.com/mvp-ai-lab/FreeScale.git
cd FreeScale

# Create and activate conda environment
conda create -n free_scale python=3.11
conda activate free_scale

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
pip install git+https://github.com/AIBluefisher/gsplat.git --no-build-isolation
pip install git+https://github.com/fraunhoferhhi/PLAS.git

# Install Gaussian Splatting submodules
cd gaussian_splatting/
pip install submodules/fused-ssim --no-build-isolation
pip install submodules/FasterGSCudaBackend
pip install submodules/MortonEncoding --no-build-isolation
cd ..