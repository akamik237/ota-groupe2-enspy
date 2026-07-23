"""
baselines.py
------------
Stratégies d'assignation "à base de règles" utilisées comme points de
comparaison face à OTA, reproduisant les heuristiques discutées dans
l'article (section 4.2, Tableau 3) :

  - Min Area  : en cas d'ambiguïté, assigner l'ancre au gt de plus petite
                surface (cf. FCOS).
  - Max IoU   : assigner au gt avec lequel l'IoU est la plus grande.
  - Min Loss  : assigner au gt qui minimise le coût cls+reg (c_fg).

Toutes ces heuristiques opèrent sur un "candidate set" (l'ensemble des
gt pour lesquels l'ancre est un candidat positif plausible, ex. Center
Prior), calculé au préalable, puis résolvent l'ambiguïté ancre par ancre,
indépendamment les unes des autres (à la différence d'OTA qui raisonne
globalement).
"""

from __future__ import annotations
import numpy as np


def candidate_mask(iou: np.ndarray, center_ok: np.ndarray, iou_thresh: float = 0.05):
    """Un couple (gt i, ancre j) est "candidat positif" si l'ancre est
    dans le rayon du Center Prior ET a une IoU non négligeable avec ce gt.
    Retourne un masque booléen (m, n)."""
    return center_ok & (iou > iou_thresh)


def assign_min_area(iou: np.ndarray, gt_areas: np.ndarray, cand: np.ndarray) -> np.ndarray:
    m, n = iou.shape
    assign = np.full(n, -1, dtype=int)
    for j in range(n):
        candidates = np.where(cand[:, j])[0]
        if len(candidates) == 0:
            continue
        # plus petite surface parmi les candidats
        best = candidates[np.argmin(gt_areas[candidates])]
        assign[j] = best
    return assign


def assign_max_iou(iou: np.ndarray, cand: np.ndarray) -> np.ndarray:
    m, n = iou.shape
    assign = np.full(n, -1, dtype=int)
    for j in range(n):
        candidates = np.where(cand[:, j])[0]
        if len(candidates) == 0:
            continue
        best = candidates[np.argmax(iou[candidates, j])]
        assign[j] = best
    return assign


def assign_min_loss(cost_fg: np.ndarray, cand: np.ndarray) -> np.ndarray:
    m, n = cost_fg.shape
    assign = np.full(n, -1, dtype=int)
    for j in range(n):
        candidates = np.where(cand[:, j])[0]
        if len(candidates) == 0:
            continue
        best = candidates[np.argmin(cost_fg[candidates, j])]
        assign[j] = best
    return assign


def count_ambiguous_heuristic(iou: np.ndarray, cand: np.ndarray) -> int:
    """Nombre d'ancres qui sont candidates positives pour >= 2 gt
    simultanément (définition de l'ambiguïté pour les heuristiques à base
    de règles, cf. section 4.2 de l'article)."""
    n_candidates_per_anchor = cand.sum(axis=0)
    return int(np.sum(n_candidates_per_anchor >= 2))
