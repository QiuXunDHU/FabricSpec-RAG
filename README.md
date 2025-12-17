# FabricSpec-RAG

## A Knowledge Graph-Augmented Seq2Seq Framework for Quantitative Analysis of Complex Textile Blends from NIR Spectra

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/FabricSpec-RAG/graphs/commit-activity)

> **Repository for the paper:** "FabricSpec-RAG: A Knowledge Graph-Augmented Seq2Seq Framework for Quantitative Analysis of Complex Textile Blends from NIR Spectra"

---

## 🚧 Code Release Status / 代码发布状态

**[English]**
This repository will contain the official PyTorch implementation of **FabricSpec-RAG**.
To ensure the integrity of the blind review process and comply with the publishing policies, the source code and pre-trained models are currently withheld.

**The full source code, datasets, and training scripts will be made publicly available immediately upon the acceptance of the manuscript.**

**[中文]**
本仓库是论文 **FabricSpec-RAG** 的官方 PyTorch 实现代码库。
为了保证双盲评审过程的公正性并遵守出版政策，源代码和预训练模型目前暂未公开。

**所有源代码、数据集和训练脚本将在稿件被正式录用后立即开源。**

---

## 📄 Abstract / 摘要

**FabricSpec-RAG** is a novel framework designed for the quantitative analysis of waste textile blends using Near-Infrared (NIR) spectroscopy. Unlike traditional regression methods, we reformulate the task as a sequence-to-sequence (Seq2Seq) generation problem.

Key innovations include:
1.  **Multi-Task Tokenization**: Decoupling component identification from ratio regression.
2.  **RAG Mechanism**: Retrieving external chemical knowledge (Knowledge Graph) to guide the generation process.
3.  **Evolutionary Feedback**: A reflection agent that iteratively refines the knowledge base.

Extensive experiments on a dataset of 1,898 spectra covering 64 textile categories demonstrate that FabricSpec-RAG significantly outperforms traditional chemometrics (PLS) and deep learning baselines (1D-CNN, ResNet).

---

## 🖼️ Architecture / 模型架构

*(Ideally, place your `net.png` or `overall_architecture.png` here to show the reviewers what the code is about)*

![Model Architecture](figures/net.png)

*Figure: Overview of the FabricSpec-RAG architecture, featuring the Multi-Scale 1D-CNN Encoder, Retrieval-Augmented Generation (RAG) fusion, and the Transformer-based Decoder.*

---

## 📅 Coming Soon / 即将发布的内容

Once released, this repository will include:
- [ ] **Data Preprocessing**: Scripts for NIR spectral smoothing, SNV, and MSC correction.
- [ ] **Knowledge Graph**: The constructed textile fiber KG and retrieval modules.
- [ ] **Model Architecture**: Full PyTorch implementation of the Encoder, RAG-Fusion, and Decoder.
- [ ] **Training Scripts**: Single-GPU and Multi-GPU training loops with Evolutionary Feedback.
- [ ] **Pre-trained Models**: Checkpoints achieving the SOTA results reported in the paper.

---

## 📧 Contact / 联系方式

If you have any questions regarding the paper or the upcoming code release, please feel free to contact:

* **Xun Qiu**: [Your Email Here]
* **Jie Zhang**: mezhangjie@dhu.edu.cn
* **Gang Wang**: gwf8707@dhu.edu.cn

---

## 🖊️ Citation

If you find this work useful for your research, please consider citing our paper (BibTeX entry will be updated upon publication):

```bibtex
@article{FabricSpecRAG2025,
  title={FabricSpec-RAG: A Knowledge Graph-Augmented Seq2Seq Framework for Quantitative Analysis of Complex Textile Blends from NIR Spectra},
  author={Qiu, Xun and Zhang, Jie and Wang, Gang and Lyu, Youlong},
  journal={Under Review},
  year={2025}
}
