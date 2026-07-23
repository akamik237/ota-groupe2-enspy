"""
test_geometry.py
------------------
Test de correction élémentaire de la fonction `box_iou` (utilisée pour le
coût de régression, Eq. 2 de l'article) sur des cas géométriques connus.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ot_assignment import box_iou


def test_identical_boxes():
    b = np.array([[0., 0., 10., 10.]])
    iou = box_iou(b, b)
    assert np.isclose(iou[0, 0], 1.0), "IoU de deux boîtes identiques doit être 1.0"
    print("[OK] IoU(boîte, elle-même) = 1.0")


def test_disjoint_boxes():
    b1 = np.array([[0., 0., 5., 5.]])
    b2 = np.array([[10., 10., 15., 15.]])
    iou = box_iou(b1, b2)
    assert np.isclose(iou[0, 0], 0.0), "IoU de boîtes disjointes doit être 0.0"
    print("[OK] IoU(boîtes disjointes) = 0.0")


def test_known_half_overlap():
    # b1 = [0,0,10,10] (aire 100), b2 = [5,0,15,10] (aire 100)
    # intersection = [5,0,10,10] (aire 50) ; union = 100+100-50 = 150
    # IoU attendu = 50/150 = 1/3
    b1 = np.array([[0., 0., 10., 10.]])
    b2 = np.array([[5., 0., 15., 10.]])
    iou = box_iou(b1, b2)
    expected = 50.0 / 150.0
    assert np.isclose(iou[0, 0], expected, atol=1e-9), \
        f"IoU attendu {expected}, obtenu {iou[0,0]}"
    print(f"[OK] IoU(chevauchement connu) = {iou[0,0]:.6f} (attendu {expected:.6f})")


def test_symmetry():
    rng = np.random.default_rng(0)
    b1 = rng.uniform(0, 50, size=(5, 2))
    b1 = np.concatenate([b1, b1 + rng.uniform(1, 20, size=(5, 2))], axis=1)
    b2 = rng.uniform(0, 50, size=(4, 2))
    b2 = np.concatenate([b2, b2 + rng.uniform(1, 20, size=(4, 2))], axis=1)
    iou_12 = box_iou(b1, b2)
    iou_21 = box_iou(b2, b1)
    assert np.allclose(iou_12, iou_21.T), "IoU doit être symétrique (IoU(A,B)=IoU(B,A))"
    print("[OK] Symétrie de l'IoU vérifiée sur des boîtes aléatoires.")


if __name__ == "__main__":
    test_identical_boxes()
    test_disjoint_boxes()
    test_known_half_overlap()
    test_symmetry()
    print("\nTous les tests géométriques sont passés avec succès.")
