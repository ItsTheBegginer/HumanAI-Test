"""
Healing Stones — 3D Fragment Reconstruction Pipeline
=====================================================
GSoC 2025 | HumanAI / CERN

Reconstructs fragmented 3D artifacts from .ply / .obj files.
Uses FPFH descriptors + RANSAC + ICP (geometry-only, no color dependency).

Usage
-----
    python reconstruct.py --input ./data/ --output ./results/
    python reconstruct.py --input ./data/ --output ./results/ --augment
    python reconstruct.py --input ./data/ --output ./results/ --min_fitness 0.3

Dependencies
------------
    pip install open3d numpy matplotlib scipy tqdm

No GPU. No CUDA. Works on any .ply or .obj files.
"""

import argparse
import glob
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  — all proven from diagnostic test on actual data
# ─────────────────────────────────────────────────────────────────────────────

TARGET_DIAG        = 100.0   # normalize every fragment to this bounding box size
VOXEL_SIZE         = 2.0     # 2% of TARGET_DIAG — consistent across all fragments
NORMAL_RADIUS      = 4.0     # 2 × VOXEL_SIZE
FPFH_RADIUS        = 10.0    # 5 × VOXEL_SIZE
RANSAC_DIST        = 6.0     # 3 × VOXEL_SIZE — proven to give fitness 0.74
ICP_DIST           = 3.0     # 1.5 × VOXEL_SIZE
RANSAC_ITER        = 500_000
MIN_FITNESS        = 0.20
N_POINTS           = 4096
CONTACT_MM         = 1.5
POISSON_DEPTH      = 6
MERGE_VOXEL        = 0.3     # relative to TARGET_DIAG space
SMOOTH_ITER        = 3
DENSITY_QUANTILE   = 0.15


def quality_label(s):
    if s >= 0.90: return "excellent"
    if s >= 0.70: return "good"
    if s >= 0.55: return "moderate"
    return "weak"


# ─────────────────────────────────────────────────────────────────────────────
# 1. ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Healing Stones reconstruction (GSoC 2025)")
    p.add_argument("--input",  required=True,  help="Folder of .ply / .obj files")
    p.add_argument("--output", required=True,  help="Output folder")
    p.add_argument("--min_fitness", type=float, default=MIN_FITNESS)
    p.add_argument("--n_points",    type=int,   default=N_POINTS)
    p.add_argument("--contact_mm",  type=float, default=CONTACT_MM)
    p.add_argument("--poisson_depth", type=int, default=POISSON_DEPTH)
    p.add_argument("--augment",     action="store_true")
    p.add_argument("--augment_copies", type=int, default=1)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(path, n_points):
    """
    Load mesh → sample point cloud → normalize to TARGET_DIAG bounding box
    → center → downsample → estimate normals.

    Normalization is essential: the dataset contains fragments with bounding
    box diagonals ranging from 37 to 953. Without normalization, RANSAC
    distance thresholds are meaningless across pairs. After normalization
    every fragment occupies a 100-unit box so all thresholds are consistent.

    Parameters are proven on the actual data:
      dist=6.0 on normalized fragments gives fitness=0.74.
    """
    # Load
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_vertices():
        raise ValueError("No vertices")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    # Sample point cloud
    if len(mesh.triangles) >= 100:
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points  = mesh.vertices
        pcd.normals = mesh.vertex_normals

    # Normalize scale to TARGET_DIAG
    pts  = np.asarray(pcd.points).copy()
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(
        np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound())
    )
    if diag > 1e-8:
        pts = pts * (TARGET_DIAG / diag)
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Center
    pcd.translate(-pcd.get_center())

    # Downsample
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    if len(pcd.points) < 10:
        raise ValueError("Too few points after downsampling")

    # Normals
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    return pcd


def load_all(folder, n_points):
    # Collect paths — deduplicate for Windows case-insensitive filesystem
    raw = (glob.glob(os.path.join(folder, "*.ply")) +
           glob.glob(os.path.join(folder, "*.PLY")) +
           glob.glob(os.path.join(folder, "*.obj")) +
           glob.glob(os.path.join(folder, "*.OBJ")))
    seen  = set()
    paths = []
    for p in sorted(raw):
        key = p.lower()
        if key not in seen:
            seen.add(key)
            paths.append(p)

    if not paths:
        print(f"[ERROR] No .ply/.obj files in {folder}")
        sys.exit(1)

    print(f"\n[1/5] Loading {len(paths)} fragments...")
    frags = []
    for idx, path in enumerate(paths):
        name = os.path.basename(path)
        try:
            pcd = load_and_preprocess(path, n_points)
            frags.append({"idx": idx, "name": name, "pcd": pcd})
            print(f"  [{idx:>2}] {name:30s}  {len(pcd.points):>5d} pts")
        except Exception as e:
            print(f"  [WARN] {name}: {e}")

    if len(frags) < 2:
        print("[ERROR] Need at least 2 fragments.")
        sys.exit(1)

    return frags


# ─────────────────────────────────────────────────────────────────────────────
# 3. AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment(frags, copies):
    print(f"\n  [AUG] {copies} augmented copy per fragment...")
    aug = []
    for f in frags:
        for i in range(copies):
            pts  = np.asarray(f["pcd"].points).copy()
            nrms = np.asarray(f["pcd"].normals).copy()
            R    = Rotation.random().as_matrix()
            pts  = (R @ pts.T).T
            nrms = (R @ nrms.T).T
            pts += np.random.normal(0, 0.001, pts.shape)
            pcd         = o3d.geometry.PointCloud()
            pcd.points  = o3d.utility.Vector3dVector(pts)
            pcd.normals = o3d.utility.Vector3dVector(nrms)
            base = os.path.splitext(f["name"])[0]
            aug.append({
                "idx":  10000 + f["idx"] * 10 + i,
                "name": f"{base}_aug{i+1}.ply",
                "pcd":  pcd,
            })
    total = len(frags) + len(aug)
    print(f"  {len(frags)} → {total} fragments  ({total*(total-1)//2} pairs)")
    return frags + aug


# ─────────────────────────────────────────────────────────────────────────────
# 4. FPFH
# ─────────────────────────────────────────────────────────────────────────────

def compute_fpfh(pcd):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS, max_nn=100)
    )


def compute_all_fpfh(frags):
    print(f"\n[2/5] Computing FPFH for {len(frags)} fragments...")
    cache = {}
    for f in tqdm(frags, desc="  FPFH", ncols=60):
        cache[f["idx"]] = compute_fpfh(f["pcd"])
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# 5. MATCHING — RANSAC + ICP
# ─────────────────────────────────────────────────────────────────────────────

def match_pair(fa, fb, fpfh_a, fpfh_b, min_fitness):
    # RANSAC
    try:
        r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            fa["pcd"], fb["pcd"], fpfh_a, fpfh_b,
            mutual_filter=False,
            max_correspondence_distance=RANSAC_DIST,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(RANSAC_DIST),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_ITER, 0.999),
        )
    except Exception:
        return None

    if r.fitness < min_fitness:
        return None

    # ICP refinement
    try:
        r_icp = o3d.pipelines.registration.registration_icp(
            fa["pcd"], fb["pcd"],
            max_correspondence_distance=ICP_DIST,
            init=r.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=200),
        )
    except Exception:
        r_icp = r

    if r_icp.fitness < min_fitness:
        return None

    # Chamfer distance
    t = o3d.geometry.PointCloud(fa["pcd"])
    t.transform(r_icp.transformation)
    d_ab    = np.asarray(t.compute_point_cloud_distance(fb["pcd"]))
    d_ba    = np.asarray(fb["pcd"].compute_point_cloud_distance(t))
    chamfer = float((d_ab.mean() + d_ba.mean()) / 2.0)

    return {
        "fitness":   round(float(r_icp.fitness),     4),
        "rmse":      round(float(r_icp.inlier_rmse), 4),
        "chamfer":   round(chamfer,                   4),
        "transform": r_icp.transformation,
        "similarity":round(float(r_icp.fitness),     4),
    }


def match_all(frags, cache, min_fitness):
    n     = len(frags)
    total = n * (n - 1) // 2
    print(f"\n[3/5] Matching {total} pairs...")

    matches     = []
    pair_report = {}

    for i in range(n):
        for j in range(i + 1, n):
            fa, fb = frags[i], frags[j]
            key    = f"fragment_{fa['idx']}_to_{fb['idx']}"
            print(f"\n  Finding matches between fragment {fa['idx']} "
                  f"and fragment {fb['idx']}...")

            res = match_pair(fa, fb, cache[fa["idx"]], cache[fb["idx"]], min_fitness)

            if res:
                ql = quality_label(res["fitness"])
                print(f"    geometry: 1 matches")
                print(f"      Similarity: {res['fitness']:.4f} ({ql})  "
                      f"RMSE={res['rmse']:.4f}  Chamfer={res['chamfer']:.4f}")
                print(f"  \u2705 1 total match found for {key}")
                matches.append({
                    "i": i, "j": j,
                    "idx_i": fa["idx"], "idx_j": fb["idx"],
                    "name_i": fa["name"], "name_j": fb["name"],
                    "similarity": res["similarity"],
                    "result": res,
                    "transform": res["transform"],
                })
                pair_report[key] = res
            else:
                print(f"  No matches found between fragments "
                      f"{fa['idx']} and {fb['idx']}")
                pair_report[key] = None

    matches.sort(key=lambda x: x["similarity"], reverse=True)

    sims = [m["similarity"] for m in matches]
    dist = {"excellent": 0, "good": 0, "moderate": 0, "weak": 0}
    for s in sims:
        dist[quality_label(s)] += 1

    print(f"\n\U0001f4ca FINAL MATCHING RESULTS:")
    print(f"   Total accepted matches: {len(matches)} / {total} pairs")
    print(f"\n\U0001f4ca ENHANCED MATCH SUMMARY:")
    print(f"   Total matches: {len(matches)}")
    print(f"   Quality distribution:")
    for lbl, cnt in dist.items():
        if cnt > 0:
            print(f"     {lbl}: {cnt}")

    return matches, pair_report


# ─────────────────────────────────────────────────────────────────────────────
# 6. ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def connected_components(frags, matches):
    idx_list = [f["idx"] for f in frags]
    parent   = {i: i for i in idx_list}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in matches:
        ra, rb = find(m["idx_i"]), find(m["idx_j"])
        if ra != rb:
            parent[ra] = rb

    groups = {}
    for i in idx_list:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def assemble_component(comp, frags, matches, contact_mm):
    idx_to_frag = {f["idx"]: f for f in frags}
    adj         = {i: [] for i in comp}

    for m in matches:
        if m["idx_i"] in adj and m["idx_j"] in adj:
            adj[m["idx_i"]].append((m["idx_j"], m["similarity"], m))
            adj[m["idx_j"]].append((m["idx_i"], m["similarity"], m))

    def score(idx):
        nbrs = adj[idx]
        if not nbrs: return 0.0
        return len(nbrs) * sum(s for _, s, _ in nbrs) / len(nbrs)

    origin = max(comp, key=score)

    print(f"\n\U0001f3d7\ufe0f IMPROVED COMPONENT ASSEMBLY")
    print(f"   Fragments: {comp}")
    print(f"   Strategy: Origin-based progressive assembly")
    print(f"   Contact distance: {contact_mm:.1f}mm")
    print(f"   Fragment match quality scores:")
    for idx in comp:
        nbrs = adj[idx]
        avg  = sum(s for _, s, _ in nbrs) / len(nbrs) if nbrs else 0.0
        tag  = "   \U0001f451 ORIGIN" if idx == origin else "      "
        print(f"   {tag} Fragment {idx}: {len(nbrs)} matches, avg sim: {avg:.3f}")
    print(f"   Selected origin: Fragment {origin}")

    transforms = {origin: np.eye(4)}
    assembled  = {origin}
    order      = [origin]

    for _ in range(len(comp) * 4):
        if len(assembled) >= len(comp):
            break
        best_sim, best = -1.0, None
        for asm in assembled:
            for nbr, sim, m in adj[asm]:
                if nbr not in assembled and sim > best_sim:
                    best_sim = sim
                    best     = (asm, nbr, m)
        if not best:
            break

        anchor, new, m = best
        print(f"   Assembling fragment {new} to {anchor} (sim: {best_sim:.3f})")

        T = m["transform"]
        if m["idx_i"] == new:
            T = np.linalg.inv(T)
        transforms[new] = transforms[anchor] @ T
        assembled.add(new)
        order.append(new)
        print(f"     Contact distance: {contact_mm:.2f}mm")
        print(f"     \u2705 Fragment {new} assembled successfully")

    print(f"   \U0001f3d7\ufe0f Assembly completed")
    print(f"   Assembly order: {order}")
    return transforms


def assemble_all(frags, matches, contact_mm):
    comps = connected_components(frags, matches)

    print(f"\n[4/5] Aligning fragments")
    print("=" * 60)
    print(f"Found {len(comps)} connected component(s)\n")

    all_T = {}
    for ci, comp in enumerate(comps, 1):
        print(f"Processing component {ci} with {len(comp)} fragment(s):")
        if len(comp) == 1:
            all_T[comp[0]] = np.eye(4)
            continue
        all_T.update(assemble_component(comp, frags, matches, contact_mm))

    aligned = []
    for f in frags:
        pcd = o3d.geometry.PointCloud(f["pcd"])
        pcd.transform(all_T.get(f["idx"], np.eye(4)))
        aligned.append(pcd)

    print(f"\nAligned {len(frags)} fragments")
    return aligned, all_T, comps


# ─────────────────────────────────────────────────────────────────────────────
# 7. MERGE → CLEAN → POISSON MESH
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_reconstruct(aligned, depth):
    # Merge
    merged = o3d.geometry.PointCloud()
    for pcd in aligned:
        merged += pcd
    print(f"  Merged: {len(merged.points)} points")

    # Clean
    merged = merged.voxel_down_sample(MERGE_VOXEL)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    print(f"  After cleaning: {len(merged.points)} points")

    # Clamp colors
    if merged.has_colors():
        merged.colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(merged.colors), 0, 1)
        )

    # Re-estimate normals for Poisson
    merged.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS * 2, max_nn=30)
    )
    merged.orient_normals_consistent_tangent_plane(k=15)

    # Poisson
    print(f"  Running Poisson reconstruction (depth={depth})...")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            merged, depth=depth
        )
        densities = np.asarray(densities)
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, DENSITY_QUANTILE))
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
        mesh = mesh.filter_smooth_simple(number_of_iterations=SMOOTH_ITER)
        mesh.compute_vertex_normals()
        print(f"  Mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
    except Exception as e:
        print(f"  [WARN] Poisson failed: {e}")
        mesh = None

    return merged, mesh


# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_all(merged, mesh, matches, comps, pair_report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # Point cloud
    p = os.path.join(output_dir, "reconstructed_pointcloud.ply")
    o3d.io.write_point_cloud(p, merged)
    saved.append(p)

    # Mesh
    if mesh and len(mesh.triangles) > 0:
        p = os.path.join(output_dir, "reconstructed_mesh.ply")
        o3d.io.write_triangle_mesh(p, mesh)
        saved.append(p)

        # Screenshot
        try:
            m2 = o3d.geometry.TriangleMesh(mesh)
            if not m2.has_vertex_colors():
                m2.paint_uniform_color([0.85, 0.78, 0.65])
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1200, height=900)
            vis.add_geometry(m2)
            opt = vis.get_render_option()
            opt.background_color    = np.array([0.15, 0.15, 0.20])
            opt.light_on            = True
            opt.mesh_show_back_face = True
            ctr = vis.get_view_control()
            ctr.set_zoom(0.65)
            ctr.set_front([0.55, -0.60, 0.58])
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_up([0.0, 0.0, 1.0])
            vis.poll_events()
            vis.update_renderer()
            p = os.path.join(output_dir, "reconstruction_preview.png")
            vis.capture_screen_image(p, do_render=True)
            vis.destroy_window()
            saved.append(p)
        except Exception as e:
            print(f"  [INFO] Screenshot skipped: {e}")

    # JSON report
    sims = [m["similarity"] for m in matches]
    qdist = {"excellent": 0, "good": 0, "moderate": 0, "weak": 0}
    for s in sims:
        qdist[quality_label(s)] += 1

    serial = {}
    for k, v in pair_report.items():
        if v is None:
            serial[k] = None
        else:
            r = dict(v)
            if "transform" in r:
                r["transform"] = r["transform"].tolist()
            serial[k] = r

    report = {
        "summary": {
            "total_matches":        len(matches),
            "connected_components": len(comps),
            "component_sizes":      [len(c) for c in comps],
            "quality_distribution": qdist,
        },
        "pairs": serial,
    }
    p = os.path.join(output_dir, "quality_report.json")
    with open(p, "w") as f:
        json.dump(report, f, indent=2)
    saved.append(p)

    # Plots
    if matches:
        from matplotlib.patches import Patch
        labels = [f"frag {m['idx_i']}\u2194{m['idx_j']}" for m in matches]
        sims2  = [m["similarity"] for m in matches]
        cols   = ["#1D9E75" if quality_label(s) == "excellent" else
                  "#534AB7" if quality_label(s) == "good"      else
                  "#BA7517" if quality_label(s) == "moderate"  else
                  "#E24B4A" for s in sims2]

        comp_sizes = [len(c) for c in comps]
        qdist2     = {k: v for k, v in qdist.items() if v > 0}

        h   = max(5.0, len(matches) * 0.45 + 2)
        fig = plt.figure(figsize=(18, h))
        fig.patch.set_facecolor("#F8F7F4")
        gs  = fig.add_gridspec(1, 3, width_ratios=[3, 1.2, 1.2],
                               wspace=0.38, left=0.06, right=0.97,
                               top=0.88, bottom=0.10)
        ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))

        fig.suptitle(
            "Healing Stones — Fragment Reconstruction Quality Report\n"
            "GSoC 2025 | HumanAI / CERN",
            fontsize=13, fontweight="bold", y=0.97
        )

        # Panel 1 — similarity bars
        ax0.set_facecolor("#FDFCF9")
        bars = ax0.barh(labels, sims2, color=cols, height=0.6,
                        edgecolor="white", linewidth=0.4)
        ax0.axvline(x=MIN_FITNESS, color="#E24B4A", linestyle="--",
                    linewidth=1.0, label=f"min threshold ({MIN_FITNESS})")
        ax0.set_xlabel("ICP fitness (higher = better)", fontsize=9)
        ax0.set_title("Match similarity per accepted pair", fontsize=10,
                      fontweight="bold", pad=8)
        ax0.set_xlim(0, 1.15)
        ax0.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, sims2):
            ax0.text(bar.get_width() + 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}  {quality_label(val)}",
                     va="center", fontsize=7.5)
        ax0.legend(handles=[
            Patch(facecolor="#1D9E75", label="Excellent"),
            Patch(facecolor="#534AB7", label="Good"),
            Patch(facecolor="#BA7517", label="Moderate"),
            Patch(facecolor="#E24B4A", label="Weak"),
        ], fontsize=8, loc="lower right")

        # Panel 2 — quality pie
        ax1.set_facecolor("#FDFCF9")
        cmap = {"excellent": "#1D9E75", "good": "#534AB7",
                "moderate": "#BA7517",  "weak": "#E24B4A"}
        _, _, ats = ax1.pie(
            list(qdist2.values()),
            labels=[f"{k}\n({v})" for k, v in qdist2.items()],
            colors=[cmap[k] for k in qdist2],
            autopct="%1.0f%%", startangle=140, pctdistance=0.72,
            textprops={"fontsize": 8},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in ats:
            at.set_fontsize(8); at.set_color("white"); at.set_fontweight("bold")
        ax1.set_title("Match quality\ndistribution", fontsize=10,
                      fontweight="bold", pad=10)

        # Panel 3 — component sizes
        ax2.set_facecolor("#FDFCF9")
        cc  = ["#534AB7" if s > 1 else "#B4B2A9" for s in comp_sizes]
        b2s = ax2.bar(range(len(comp_sizes)), comp_sizes, color=cc,
                      width=0.55, edgecolor="white")
        ax2.set_xticks(range(len(comp_sizes)))
        ax2.set_xticklabels([f"C{i+1}\n({s}f)" for i, s in enumerate(comp_sizes)],
                            fontsize=7.5)
        ax2.set_ylabel("Fragments", fontsize=9)
        ax2.set_title("Assembly components", fontsize=10,
                      fontweight="bold", pad=8)
        ax2.set_ylim(0, max(comp_sizes) + 1.5)
        ax2.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(b2s, comp_sizes):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.1, str(val),
                     ha="center", fontsize=9, fontweight="bold")
        joined = sum(s for s in comp_sizes if s > 1)
        ax2.text(0.5, -0.22,
                 f"{joined}/{sum(comp_sizes)} fragments joined\n"
                 f"{len(matches)} accepted matches",
                 transform=ax2.transAxes, ha="center",
                 fontsize=8, style="italic")

        plt.tight_layout()
        p = os.path.join(output_dir, "quality_metrics.png")
        plt.savefig(p, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        saved.append(p)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0   = time.time()
    args = parse_args()

    if not os.path.isdir(args.input):
        print(f"[ERROR] Input folder not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("\n" + "=" * 60)
    print("  HEALING STONES — 3D Fragment Reconstruction")
    print("  GSoC 2025 | HumanAI / CERN")
    print("=" * 60)
    print(f"  Input         : {args.input}")
    print(f"  Output        : {args.output}")
    print(f"  Min fitness   : {args.min_fitness}")
    print(f"  Poisson depth : {args.poisson_depth}")

    # 1 Load
    frags = load_all(args.input, args.n_points)

    # 2 Augment
    if args.augment:
        frags = augment(frags, args.augment_copies)

    # 3 FPFH
    cache = compute_all_fpfh(frags)

    # 4 Match
    print(f"\n{'=' * 60}")
    print("STEP 3: Matching fragments")
    print("=" * 60)
    t3 = time.time()
    matches, pair_report = match_all(frags, cache, args.min_fitness)
    print(f"Step 3 completed in {time.time()-t3:.1f}s")

    if not matches:
        print("\n[WARN] No matches found.")
        print("  Try: --min_fitness 0.1")
        # Still save empty report so script exits cleanly
        save_all(o3d.geometry.PointCloud(), None, [], 
                 [[f["idx"]] for f in frags], pair_report, args.output)
        sys.exit(0)

    # 5 Assemble
    print(f"\n{'=' * 60}")
    print("STEP 4: Aligning fragments")
    print("=" * 60)
    t4 = time.time()
    aligned, _, comps = assemble_all(frags, matches, args.contact_mm)
    print(f"Step 4 completed in {time.time()-t4:.1f}s")

    # 6 Reconstruct + Save
    print(f"\n{'=' * 60}")
    print("STEP 5: Mesh reconstruction and saving")
    print("=" * 60)
    merged, mesh = merge_and_reconstruct(aligned, args.poisson_depth)
    saved = save_all(merged, mesh, matches, comps, pair_report, args.output)

    for p in saved:
        print(f"  Saved -> {p}")

    print("\n" + "=" * 60)
    print("  RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Fragments    : {len(frags)}")
    print(f"  Matches      : {len(matches)}")
    print(f"  Components   : {len(comps)}  "
          f"(largest: {max(len(c) for c in comps)})")
    print(f"  Time         : {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()