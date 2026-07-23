# OTA — Groupe 2 ENSPY (Computer Vision)

**Article :** Ge et al., *OTA: Optimal Transport Assignment for Object Detection*, CVPR 2021.

**Membres :** Bissog Samuel Durand (20P125), BOURMEKE KIMAKA Junior Dimitri (20P232), Keumouo Tadaha Diroil James (20P172).

## Contenu

| Dossier | Description |
|---------|-------------|
| `code/` | Implémentation OTA (Sinkhorn-Knopp, coût cls+reg, Dynamic k, baselines) |
| `figures/` | Cartes d'assignation et courbe ambiguïté vs rayon |
| `tables/` | CSV résultats expérience synthétique |
| `report/` | Rapport scientifique + carnet de lecture (LaTeX/PDF) |

## Installation

```bash
pip install -r requirements.txt
```

## Reproduire l'implémentation

```bash
python demo.py
```

Ou étape par étape :

```bash
cd code
python tests/test_geometry.py
python tests/test_sinkhorn.py
python experiment_ambiguous_anchors.py
```

## Résultats

- **Validation Sinkhorn-Knopp** vs LP exact (`scipy.optimize.linprog`) : écart relatif 0,16–0,73 % pour $\gamma=0.1$.
- **Expérience synthétique** (2 objets qui se chevauchent, 625 ancres) : comparaison OTA vs Min Area / Max IoU / Min Loss.
- Figures : `figures/assignment_maps.png`, `figures/ambiguous_vs_radius.png`
- Tableau : `tables/ambiguous_anchor_counts.csv`

## Rapport

- [`report/rapport.pdf`](report/rapport.pdf) — rapport scientifique (~13 pages)
- [`report/carnet_lecture.pdf`](report/carnet_lecture.pdf) — carnet de lecture (1–2 pages)

Compilation LaTeX :

```bash
cd report
pdflatex rapport.tex && bibtex rapport && pdflatex rapport.tex && pdflatex rapport.tex
pdflatex carnet_lecture.tex && pdflatex carnet_lecture.tex
```

## Dépôt GitHub

**https://github.com/akamik237/ota-groupe2-enspy**

Professeur (`htapamo`) : accès en lecture invité sur le dépôt (repo public + invitation collaborateur).


Nous **ne reproduisons pas** l'entraînement FCOS-ResNet-50 sur COCO (AP 40,7 % de l'article) : pas de GPU / PyTorch / accès COCO dans notre environnement. Le dépôt se concentre sur l'**algorithme d'assignation OTA** lui-même, validé numériquement et sur un scénario synthétique contrôlé. Voir la section Analyse critique du rapport.

## Référence

- Article : [OTA CVPR 2021](https://arxiv.org/abs/2103.14259)
- Code officiel auteurs : [Megvii-BaseDetection/OTA](https://github.com/Megvii-BaseDetection/OTA)
