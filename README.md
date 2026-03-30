# Healing Stones — 3D Fragment Reconstruction
**GSoC 2025 | HumanAI / CERN**

Reconstructs a fragmented 3D artifact from a folder of `.ply` or `.obj` files using FPFH descriptors, RANSAC-based correspondence matching, and ICP refinement.

---

## Setup

```bash
pip install -r requirements.txt
```

No GPU required. Runs on any CPU. Tested on Python 3.8.

---

## Usage

```bash
# Basic run
python reconstruct.py --input ./data/ --output ./results/

# With data augmentation (random rotations + jitter per fragment)
python reconstruct.py --input ./data/ --output ./results/ --augment [Takes too much time!]


```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Folder containing `.ply` / `.obj` files |
| `--output` | required | Folder where outputs are saved |
| `--voxel_size` | auto | Downsampling resolution. Auto = 2% of bounding box diagonal |
| `--min_fitness` | 0.25 | Minimum RANSAC overlap ratio to accept a match |
| `--n_points` | 8192 | Points sampled per mesh surface |
| `--augment` | off | Add random-rotation + jitter copies of each fragment |
| `--augment_copies` | 3 | Number of augmented copies per fragment |

---

## Outputs

| File | Description |
|---|---|
| `results/reconstructed.ply` | Merged point cloud of all aligned fragments |
| `results/quality_report.json` | Per-pair metrics: fitness, RMSE, Chamfer distance |
| `results/quality_metrics.png` | Bar charts of all quality metrics |

---

## How it works

### 1. Load & preprocess
Every `.ply` / `.obj` file in `--input` is loaded, converted to a point cloud, centered at the origin, voxel-downsampled, and fitted with consistent surface normals.

### 2. FPFH feature extraction
Fast Point Feature Histograms (Rusu et al., 2009) compute a 33-dimensional rotation-invariant descriptor at every point based on angular relationships between surface normals. The same fracture surface produces the same descriptor regardless of fragment orientation — directly solving the randomly-rotated fragments problem.

### 3. RANSAC matching
For every unique fragment pair, RANSAC samples 4-point FPFH correspondences, computes the rigid transformation that satisfies them, counts inliers, and repeats up to 4 million times. This is a probabilistic model-fitting algorithm — the core AI component of the pipeline. Pairs below `--min_fitness` are rejected.

### 4. ICP refinement
Point-to-Plane ICP refines each coarse RANSAC alignment to sub-millimeter precision by iteratively minimising point-to-surface distances.

### 5. Assembly & output
All aligned fragments are merged into one point cloud. Statistical outliers are removed. Quality metrics (ICP fitness, RMSE, Chamfer distance) are computed per pair, saved as JSON, and plotted.

---

## Metrics

- **ICP fitness** — fraction of point pairs within correspondence distance after alignment (higher = better, max 1.0)
- **ICP inlier RMSE** — root mean squared distance of inlier pairs (lower = better)
- **Chamfer distance** — symmetric average nearest-neighbour distance between aligned fragments (lower = better)

---

## References

- Rusu et al. (2009). *Fast Point Feature Histograms (FPFH) for 3D Registration.* ICRA.
- Chen & Medioni (1992). *Object Modelling by Registration of Multiple Range Images.* Image and Vision Computing.
- Zhou et al. (2018). *Open3D: A Modern Library for 3D Data Processing.* arXiv:1801.09847.



### Reconstructed mesh

![Reconstruction preview](results_preview.png)
<img width="464" height="264" alt="image" src="https://github.com/user-attachments/assets/ab770220-4b31-4085-9057-2591486fe676" />

### Quality metrics

![Quality metrics](quality_metrics.png)
<img width="3145" height="11636" alt="image" src="https://github.com/user-attachments/assets/11e4fa7f-5fce-4ebf-a8f1-9b531d84d708" />


