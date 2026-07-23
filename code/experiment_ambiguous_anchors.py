"""
experiment_ambiguous_anchors.py
--------------------------------
Expérience synthétique reproduisant QUALITATIVEMENT le phénomène central
de l'article (Figure 1, Figure 3, Tableaux 2 et 3) : la gestion des
"ancres ambiguës" (candidates positives pour plusieurs gt à la fois), en
comparant OTA aux heuristiques Min Area / Max IoU / Min Loss.

*** Avertissement méthodologique (voir aussi le rapport, section 6) ***
Faute de GPU, d'accès à COCO et de PyTorch dans cet environnement, nous
NE reproduisons PAS l'entraînement d'un détecteur FCOS réel. À la place,
nous construisons un scénario synthétique contrôlé : deux boîtes "vérité
terrain" se chevauchant (comme sur la Figure 1 de l'article) et une
grille d'ancres, avec des coûts cls/reg synthétiques simulant un
détecteur partiellement entraîné (scores de confiance bruités,
corrélés à la distance au centre du bon objet). Cela permet de tester
et de valider fidèlement l'ALGORITHME d'assignation (ce qui est
l'objet du sujet), sans prétendre reproduire les scores d'AP sur COCO.

Sorties : figures/assignment_maps.png, figures/ambiguous_vs_radius.png,
          tables/ambiguous_anchor_counts.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ot_assignment import (
    build_cost_matrix, sinkhorn_knopp, dynamic_k_estimation,
    build_supply_vector, decode_assignment, ambiguous_anchor_mask, box_iou
)
from baselines import (
    candidate_mask, assign_min_area, assign_max_iou, assign_min_loss,
    count_ambiguous_heuristic
)

RNG = np.random.default_rng(7)


def make_scene():
    """Deux boîtes qui se chevauchent partiellement (analogue à la Figure 1
    de l'article : deux objets proches, ex. deux personnes ou une
    personne + un objet porté)."""
    gt_boxes = np.array([
        [20.0, 20.0, 60.0, 70.0],   # gt 0
        [45.0, 25.0, 85.0, 75.0],   # gt 1, chevauche gt 0
    ])
    # grille d'ancres (points, "anchor-free" comme FCOS) sur [0,100]x[0,100]
    xs = np.linspace(2, 98, 25)
    ys = np.linspace(2, 98, 25)
    gx, gy = np.meshgrid(xs, ys)
    anchor_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    box_half = 3.0  # taille fixe de la "boîte d'ancre" pour le calcul d'IoU
    anchor_boxes = np.concatenate([
        anchor_pts - box_half, anchor_pts + box_half
    ], axis=1)
    return gt_boxes, anchor_pts, anchor_boxes


def synthetic_reg_quality(anchor_pts, gt_boxes, scale=18.0):
    """Simule, pour CHAQUE paire (gt, ancre), une "qualité de régression"
    dans [0,1] (utilisée à la place d'une IoU géométrique réelle) qui
    décroît avec la distance au centre du gt -- modélisant le fait qu'un
    détecteur partiellement entraîné régresse mieux les boîtes des
    ancres proches du centre de l'objet. C'est cette quantité qui joue
    le rôle de `iou` dans le reste du pipeline (coût de régression,
    Dynamic k Estimation), à la place d'une IoU géométrique triviale
    (ancres ponctuelles vs. boîtes) qui serait artificiellement faible.
    Voir avertissement méthodologique en tête de fichier.
    """
    m = gt_boxes.shape[0]
    centers = np.stack([
        (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
        (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    ], axis=1)
    quality = np.zeros((m, anchor_pts.shape[0]))
    for i in range(m):
        d = np.linalg.norm(anchor_pts - centers[i], axis=1)
        quality[i] = np.exp(-(d ** 2) / (2 * scale ** 2))
    return quality


def synthetic_pred_scores(anchor_pts, gt_boxes):
    """Simule les scores de confiance d'un détecteur partiellement
    entraîné : score élevé près du centre du gt le plus proche, décroît
    avec la distance, plus un bruit gaussien (incertitude d'entraînement).
    """
    centers = np.stack([
        (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
        (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    ], axis=1)
    dists = np.linalg.norm(anchor_pts[:, None, :] - centers[None, :, :], axis=2)
    min_dist = np.min(dists, axis=1)
    base_score = np.exp(-min_dist / 25.0)
    noise = RNG.normal(0, 0.05, size=base_score.shape)
    return np.clip(base_score + noise, 0.02, 0.98)


def run_ota(gt_boxes, anchor_pts, anchor_boxes, pred_scores, radius, reg_quality, q=15):
    cost, iou = build_cost_matrix(
        anchor_boxes, anchor_pts, gt_boxes, pred_scores,
        alpha=1.5, center_radius=radius, center_penalty=50.0,
        iou_override=reg_quality
    )
    k = dynamic_k_estimation(iou, q=q)
    n = anchor_pts.shape[0]
    s = build_supply_vector(k, n)
    d = np.ones(n)
    pi = sinkhorn_knopp(cost, s, d, gamma=0.1, n_iters=300)
    assign = decode_assignment(pi)
    amb = ambiguous_anchor_mask(pi, threshold=0.9)
    return assign, amb, iou, k


def run_heuristics(gt_boxes, anchor_pts, iou, cost_fg, radius):
    gt_centers = np.stack([
        (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
        (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    ], axis=1)
    dists = np.linalg.norm(anchor_pts[None, :, :] - gt_centers[:, None, :], axis=2)
    center_ok = np.zeros_like(iou, dtype=bool)
    for i in range(gt_boxes.shape[0]):
        order = np.argsort(dists[i])
        center_ok[i, order[:radius]] = True

    cand = candidate_mask(iou, center_ok, iou_thresh=0.3)
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    assign_area = assign_min_area(iou, gt_areas, cand)
    assign_iou = assign_max_iou(iou, cand)
    assign_loss = assign_min_loss(cost_fg, cand)
    n_amb = count_ambiguous_heuristic(iou, cand)
    return assign_area, assign_iou, assign_loss, n_amb


def plot_assignment_maps(gt_boxes, anchor_pts, assigns_dict, filename):
    fig, axes = plt.subplots(1, len(assigns_dict), figsize=(4.2 * len(assigns_dict), 4.6))
    colors = ["#4C72B0", "#DD8452", "#999999"]  # gt0, gt1, fond
    for ax, (name, assign) in zip(axes, assigns_dict.items()):
        for i, box in enumerate(gt_boxes):
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                  fill=False, edgecolor="black", linewidth=1.5,
                                  linestyle="--")
            ax.add_patch(rect)
        pos_mask = assign >= 0
        ax.scatter(anchor_pts[~pos_mask, 0], anchor_pts[~pos_mask, 1],
                   c="lightgray", s=6, label="fond")
        for i in range(gt_boxes.shape[0]):
            m = assign == i
            ax.scatter(anchor_pts[m, 0], anchor_pts[m, 1],
                       c=colors[i], s=14, label=f"gt{i}")
        ax.set_title(name)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
    axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def experiment_radius_sensitivity(gt_boxes, anchor_pts, anchor_boxes, pred_scores,
                                   reg_quality, radii=(3, 6, 10, 15, 25, 40)):
    rows = []
    for r in radii:
        assign_ota, amb_ota, iou, k = run_ota(gt_boxes, anchor_pts, anchor_boxes,
                                                pred_scores, radius=r, reg_quality=reg_quality)
        cost, _ = build_cost_matrix(anchor_boxes, anchor_pts, gt_boxes, pred_scores,
                                     alpha=1.5, center_radius=r, center_penalty=50.0,
                                     iou_override=reg_quality)
        cost_fg = cost[:-1]
        _, _, _, n_amb_heuristic = run_heuristics(gt_boxes, anchor_pts, iou, cost_fg, radius=r)
        rows.append({
            "radius_r": r,
            "n_ambiguous_OTA": int(np.sum(amb_ota)),
            "n_ambiguous_heuristics": n_amb_heuristic,
            "k_gt0": int(k[0]),
            "k_gt1": int(k[1]),
        })
        print(f"r={r:>3}: ambigües OTA={rows[-1]['n_ambiguous_OTA']:>4}   "
              f"ambigües heuristiques (candidats multiples)={n_amb_heuristic:>4}   "
              f"k=({k[0]},{k[1]})")
    return pd.DataFrame(rows)


def main():
    gt_boxes, anchor_pts, anchor_boxes = make_scene()
    pred_scores = synthetic_pred_scores(anchor_pts, gt_boxes)
    reg_quality = synthetic_reg_quality(anchor_pts, gt_boxes)

    # --- Visualisation des cartes d'assignation pour r=15 (analogue Fig. 3) ---
    r_vis = 15
    assign_ota, amb_ota, iou, k = run_ota(gt_boxes, anchor_pts, anchor_boxes,
                                            pred_scores, radius=r_vis, reg_quality=reg_quality)
    cost, _ = build_cost_matrix(anchor_boxes, anchor_pts, gt_boxes, pred_scores,
                                 alpha=1.5, center_radius=r_vis, center_penalty=50.0,
                                 iou_override=reg_quality)
    cost_fg = cost[:-1]
    assign_area, assign_iou, assign_loss, n_amb_h = run_heuristics(
        gt_boxes, anchor_pts, iou, cost_fg, radius=r_vis)

    plot_assignment_maps(gt_boxes, anchor_pts, {
        "Min Area": assign_area,
        "Max IoU": assign_iou,
        "Min Loss": assign_loss,
        "OTA": assign_ota,
    }, "../figures/assignment_maps.png")

    print(f"\n(r={r_vis}) Ancres ambiguës -- OTA: {int(np.sum(amb_ota))}, "
          f"candidates multiples (heuristiques): {n_amb_h}\n")
    print(f"Dynamic k estimé : gt0={k[0]}, gt1={k[1]}\n")

    # --- Sensibilité au rayon r (analogue Tableau 2) ---
    df = experiment_radius_sensitivity(gt_boxes, anchor_pts, anchor_boxes, pred_scores, reg_quality)
    df.to_csv("../tables/ambiguous_anchor_counts.csv", index=False)

    plt.figure(figsize=(6, 4.2))
    plt.plot(df["radius_r"], df["n_ambiguous_OTA"], marker="^", label="OTA (max $\\pi^*_j$ < 0.9)")
    plt.plot(df["radius_r"], df["n_ambiguous_heuristics"], marker="o",
             label="Heuristiques (candidates multiples)")
    plt.xlabel("Rayon du Center Prior $r$ (nombre d'ancres les plus proches)")
    plt.ylabel("Nombre d'ancres ambiguës")
    plt.title("Sensibilité du nombre d'ancres ambiguës à $r$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../figures/ambiguous_vs_radius.png", dpi=150)
    plt.close()

    print("Résultats sauvegardés : figures/assignment_maps.png, "
          "figures/ambiguous_vs_radius.png, tables/ambiguous_anchor_counts.csv")


if __name__ == "__main__":
    main()
