# Physics-Informed Neural Networks (PINNs) - Équation de Burgers

Implémentation PyTorch d'un réseau de neurones informé par la physique (PINN) pour résoudre l'équation de Burgers.

## Description

Ce projet implémente l'approche PINN développée par Raissi et al. (2019) pour résoudre des équations aux dérivées partielles (EDP) non linéaires. L'équation de Burgers sert de cas d'étude :

$$u_t + u \cdot u_x - \nu \cdot u_{xx} = 0$$

avec les conditions initiales et aux limites appropriées.

## Structure du Projet

```
├── src/
│   ├── burger_eq.py      # Résidus PDE et conditions aux limites
│   ├── networks.py       # Architecture PINN (forward/inverse)
│   ├── utils.py          # Utilitaires (génération de données, métriques)
│   └── viz.py            # Fonctions de visualisation
├── data/
│   └── burgers_shock.mat # Données de référence
├── demo.ipynb            # Notebook de démonstration complet
└── pinn_errors_*.csv     # Résultats d'expériences
```

## Fonctionnalités

- **PINN Forward** : Résolution de l'EDP avec données de référence limitées
- **PINN Inverse** : Identification du paramètre physique (viscosité ν)
- **Analyses paramétriques** :
  - Impact du nombre de points de collocation
  - Influence des poids de la fonction de perte
  - Effet de l'architecture du réseau
- **Visualisations** : Comparaison des solutions, erreurs, et convergence
## Installation

```bash
pip install torch numpy scipy matplotlib pyDOE tqdm pandas
```

## Utilisation

Lancez le notebook [demo.ipynb](demo.ipynb) pour explorer :
1. Configuration du problème et visualisation des données
2. Entraînement du PINN forward
3. Résolution du problème inverse
4. Études paramétriques et analyses de sensibilité
5. Comparaison avec approches hybrides

## Références

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125). *Journal of Computational Physics*, 378, 686-707.

## Résultats

Les expériences comparent l'impact de différents hyperparamètres sur l'erreur L2 relative, avec résultats sauvegardés dans les fichiers CSV.

## Extension

En plus des analyses d'architectures et de la taille des données, j'ai choisi de rajouter une autre analyse, celle de la loss, puisque c'est là que réside le changement apporter par les PINNs.
