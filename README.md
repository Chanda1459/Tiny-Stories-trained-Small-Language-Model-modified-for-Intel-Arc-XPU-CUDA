# Small Language Model from Scratch on Intel Arc XPU

An experimental Small Language Model (SLM) implementation developed using PyTorch, adapted to run on Intel Arc GPUs through the PyTorch XPU backend.

## Project Overview

This project explores the development and training of a Small Language Model from scratch, with a focus on adapting the training pipeline to Intel GPU hardware.

## Objectives

- Implement a Small Language Model architecture.
- Understand the underlying training pipeline of transformer-based language models.
- Adapt PyTorch operations for Intel Arc XPU acceleration.
- Experiment with model training and evaluation.
- Explore performance and compatibility considerations when using Intel GPU hardware.

## Project Structure

```text
SLM-Intel-Arc-XPU/
├── SLM_IntelArc_XPU.ipynb
├── Vizuara_AI_Labs_Small_Language_Model_Scratch_Final.ipynb
├── requirements.txt
├── loss_curves.png
├── README.md
└── .gitignore
```

## Hardware and Software

- GPU: Intel Arc
- Framework: PyTorch
- Acceleration: PyTorch XPU
- Language: Python
- Environment: Windows

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Ensure your PyTorch installation supports Intel XPU before running the notebooks.

## Datasets and Model Checkpoints

Large datasets and trained model checkpoints are maintained separately from the Git repository.

Instructions for obtaining these artifacts will be added here.

## Acknowledgements

This project builds upon the Small Language Model learning material by Vizuara AI Labs. The original notebook is retained for reference and attribution.

## Project Status

Experimental / In Development.