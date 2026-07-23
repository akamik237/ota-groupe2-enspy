"""
ot_assignment.py
-----------------
Implémentation fidèle de l'algorithme d'assignation de labels par
Transport Optimal (Optimal Transport Assignment, OTA), tel que décrit
dans :

  Z. Ge, S. Liu, Z. Li, O. Yoshie, J. Sun.
  "OTA: Optimal Transport Assignment for Object Detection." CVPR 2021.

Ce module implémente :
  1. La résolution du problème de transport optimal régularisé par
     entropie via l'itération de Sinkhorn-Knopp (Eq. 5-10 et Algorithme 1
     de l'article, cf. aussi l'Annexe A.1 de l'article pour la dérivation).
  2. La construction de la matrice de coût (cls + reg + center prior),
     Eq. 2-4 de l'article.
  3. L'estimation dynamique de k ("Dynamic k Estimation", section 3.3).
  4. Le décodage de l'assignation finale à partir du plan de transport.

Toutes les formules sont reproduites fidèlement (mêmes notations que
l'article dans la mesure du possible) ; les commentaires renvoient aux
équations correspondantes du papier.
"""

from __future__ import annotations
import numpy as np


def sinkhorn_knopp(cost: np.ndarray, s: np.ndarray, d: np.ndarray,
                    gamma: float = 0.1, n_iters: int = 50) -> np.ndarray:
    """Résout le problème de transport optimal régularisé par entropie
    (Eq. 5 de l'article) via l'itération de Sinkhorn-Knopp (Eq. 10).

    Paramètres
    ----------
    cost : matrice (m+1, n) des coûts unitaires de transport (dernière
           ligne = coût vers le fond/"background").
    s    : vecteur (m+1,) des quantités fournies par chaque fournisseur
           (gt_1, ..., gt_m, background).
    d    : vecteur (n,) des quantités demandées par chaque ancre
           (= 1 pour chacune, Eq. après (4)).
    gamma: intensité de la régularisation entropique (0.1 dans l'article).
    n_iters : nombre d'itérations (50 dans l'article).

    Retour
    ------
    pi : matrice (m+1, n), le plan de transport approché pi* (Eq. 11).
    """
    M = np.exp(-cost / gamma)          # Eq. 7 : M_ij = exp(-c_ij/gamma)
    u = np.ones(cost.shape[1])          # u^0 (OnesInit, ligne 12 Algo 1)
    v = np.ones(cost.shape[0])          # v^0

    for _ in range(n_iters):
        # Eq. 10 : mise à jour alternée
        Mv = M.T @ v                    # (n,)
        u = d / np.clip(Mv, 1e-12, None)
        Mu = M @ u                      # (m+1,)
        v = s / np.clip(Mu, 1e-12, None)

    pi = np.diag(v) @ M @ np.diag(u)    # Eq. 11
    return pi


def sinkhorn_residual(pi: np.ndarray, s: np.ndarray, d: np.ndarray) -> float:
    """Mesure l'écart aux contraintes marginales (diagnostic de
    convergence) : max( |somme lignes - s|, |somme colonnes - d| )."""
    row_err = np.max(np.abs(pi.sum(axis=1) - s))
    col_err = np.max(np.abs(pi.sum(axis=0) - d))
    return max(row_err, col_err)


# ----------------------------------------------------------------------
# Géométrie : IoU et coûts cls/reg synthétiques
# ----------------------------------------------------------------------
def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """IoU par paire entre deux ensembles de boîtes [x1,y1,x2,y2].
    boxes1: (n,4), boxes2: (m,4) -> retourne (n,m)."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-12, None)


def build_cost_matrix(anchor_boxes: np.ndarray, anchor_pts: np.ndarray,
                       gt_boxes: np.ndarray, pred_scores: np.ndarray,
                       alpha: float = 1.5, center_radius: float = None,
                       center_penalty: float = 100.0,
                       iou_override: np.ndarray = None):
    """Construit la matrice de coût complète c in R^{(m+1) x n}
    (Eq. 2-4 de l'article), avec Center Prior (section 3.3).

    Paramètres
    ----------
    anchor_boxes : (n,4) boîtes des ancres, pour le calcul de l'IoU (reg).
    anchor_pts   : (n,2) centres des ancres, pour le Center Prior.
    gt_boxes     : (m,4) boîtes vérité-terrain.
    pred_scores  : (n,) scores de confiance PRÉDITS par un détecteur
                   (synthétiques ici, cf. avertissement du rapport) pour
                   la vraie classe -- utilisés pour le coût cls (focal-loss
                   like) et pour le coût de fond.
    alpha        : coefficient de pondération reg (1.5 dans l'article, Eq. 2).
    center_radius: si fourni, seules les `center_radius` ancres les plus
                   proches du centre de chaque gt sont éligibles comme
                   positives (Center Prior) ; les autres reçoivent un
                   coût constant additionnel `center_penalty`.

    Retour
    ------
    cost : (m+1, n)
    iou  : (m, n)  matrice d'IoU anchor-gt (utile pour Dynamic k Estimation)
    """
    n = anchor_boxes.shape[0]
    m = gt_boxes.shape[0]

    if iou_override is not None:
        iou = iou_override                                     # (m, n) synthétique
    else:
        iou = box_iou(gt_boxes, anchor_boxes)                  # (m, n) géométrique exact

    # --- coût de régression (Eq. 2) : IoU loss = 1 - IoU ---
    c_reg = 1.0 - iou                                          # (m, n)

    # --- coût de classification (Eq. 2) : proxy focal-loss-like ---
    # On simule ici un score de confiance PRÉDIT (pred_scores) pour la
    # "bonne" classe de chaque ancre (indépendant du gt), et on calcule un
    # cross-entropy focal-loss-like coût pour l'associer à chaque gt.
    # (Simplification assumée et documentée dans le rapport : dans
    # l'article, pred_scores dépend du réseau réellement entraîné.)
    eps = 1e-8
    p = np.clip(pred_scores, eps, 1 - eps)                     # (n,)
    gamma_focal = 2.0
    focal = -((1 - p) ** gamma_focal) * np.log(p)              # (n,)
    c_cls = np.tile(focal[None, :], (m, 1))                    # (m, n)

    c_fg = c_cls + alpha * c_reg                                # Eq. 2

    # --- Center Prior (section 3.3) ---
    if center_radius is not None:
        gt_centers = np.stack([
            (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0,
            (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
        ], axis=1)                                             # (m, 2)
        dists = np.linalg.norm(
            anchor_pts[None, :, :] - gt_centers[:, None, :], axis=2
        )                                                       # (m, n)
        for i in range(m):
            order = np.argsort(dists[i])
            not_close = order[center_radius:]
            c_fg[i, not_close] += center_penalty

    # --- coût de fond (background), Eq. 3 : uniquement le coût cls ---
    c_bg = -np.log(1 - p + eps)                                 # (n,) coût si l'ancre est "vraiment" fond
    # NOTE : pour une ancre qui a une vraie classe positive plausible,
    # l'assigner au fond coûte cher si son score prédit p est élevé.

    cost = np.vstack([c_fg, c_bg[None, :]])                     # (m+1, n)
    return cost, iou


def dynamic_k_estimation(iou: np.ndarray, q: int = 5) -> np.ndarray:
    """Section 3.3 : pour chaque gt, on somme les q plus grandes valeurs
    d'IoU parmi toutes les ancres pour estimer le nombre de labels
    positifs que ce gt doit fournir (arrondi à l'entier supérieur, borné
    à au moins 1).
    """
    m, n = iou.shape
    k = np.zeros(m, dtype=int)
    qq = min(q, n)
    for i in range(m):
        top_q = np.sort(iou[i])[::-1][:qq]
        k[i] = max(1, int(np.round(np.sum(top_q))))
    return k


def build_supply_vector(k: np.ndarray, n_anchors: int) -> np.ndarray:
    """Eq. 4 : s_i = k_i pour i <= m, s_{m+1} = n - sum(k)."""
    total_pos = int(np.sum(k))
    total_pos = min(total_pos, n_anchors)  # sécurité si k trop grand
    s_bg = n_anchors - total_pos
    return np.concatenate([k.astype(float), [float(s_bg)]])


def decode_assignment(pi: np.ndarray) -> np.ndarray:
    """Décodage de l'assignation finale (section 3.2, après Eq. 11) :
    chaque ancre est assignée au fournisseur (gt ou fond, dernière ligne)
    qui lui transporte la plus grande quantité de labels.
    Retour : vecteur (n,) avec la valeur = indice du gt (0..m-1) ou -1
    pour le fond (dernière ligne = indice m).
    """
    m_plus_1, n = pi.shape
    m = m_plus_1 - 1
    assign = np.argmax(pi, axis=0)  # (n,), valeurs dans [0, m]
    assign = np.where(assign == m, -1, assign)
    return assign


def ambiguous_anchor_mask(pi: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Définition de l'article (section 4.2) : une ancre a_j est ambiguë
    si max_i pi*_{ij} < 0.9 (le plan de transport ne "penche" pas
    clairement vers un seul fournisseur)."""
    return np.max(pi, axis=0) < threshold
