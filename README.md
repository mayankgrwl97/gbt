# Geometry-biased Transformers for Novel View Synthesis
Geometry-biased Transformers for Novel View Synthesis

[Paper](https://arxiv.org/pdf/2301.04650.pdf)&nbsp;&nbsp;&nbsp;
[Web](https://mayankgrwl97.github.io/gbt/)&nbsp;&nbsp;&nbsp;
[Demo](https://colab.research.google.com/github/mayankgrwl97/gbt/blob/main/notebooks/gbt-demo.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg?style=for-the-badge)](https://colab.research.google.com/github/mayankgrwl97/gbt/blob/main/notebooks/gbt-demo.ipynb)

## Environment Setup
For detailed instructions refer to [SETUP.md](./SETUP.md)

## Dataset
Follow instructions from the official CO3D [repository](https://github.com/facebookresearch/co3d#download-the-dataset) to download the dataset in [this](https://github.com/facebookresearch/co3d#dataset-format) format.

## Training
(Coming Soon)

## Inference
### Checkpoints
Download pre-trained checkpoints from this [link](https://drive.google.com/file/d/1eHeNba_qlsM-7iEiIlZw9XH9-VXqem7T/view?usp=sharing). Extract contents inside the repository base directory. Alternatively, run the following commands from terminal.
```bash
pip install gdown
gdown 1eHeNba_qlsM-7iEiIlZw9XH9-VXqem7T
unzip runs.zip
rm runs.zip
```

Verify that the extracted checkpoints are of the following structure.

```bash
gbt/runs/co3dv2/cat_agnostic/
|-- gbt
|   `-- latest.pt
`-- gbt_nb
    `-- latest.pt
```

### Commands

Run `GBT` model trained on 10 categories (category agnostic)
```bash
python scripts/infer.py --config-path configs/cat_agnostic_gbt.yaml --dataset-path /path/to/co3d/dataset --category donut
```

Run `GBT-nb` (no geometric bias) model trained on 10 categories (category agnostic)
```bash
python scripts/infer.py --config-path configs/cat_agnostic_gbt_nb.yaml --dataset-path /path/to/co3d/dataset --category donut
```

### Output
The inference script computes average `psnr` and `lpips` metrics for objects of the specified category, and also saves individual rotating gifs for qualitative analysis.

```bash
runs/co3dv2/cat_agnostic/gbt/infer/num_views=3/donut/
|-- 198_21296_42378.gif
|-- 290_30761_58510.gif
|-- ...
`-- metrics.txt
```

<!-- ## BibTeX -->

<!-- ## Acknowledgements -->
