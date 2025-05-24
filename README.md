# Programme de Gestion de Graphes

Un outil en ligne de commande pour créer et analyser des graphes orientés ou non orientés.  
Vous pouvez construire votre graphe, l’afficher (liste d’adjacence et visualisation), et exécuter divers algorithmes et tests :

- **Construction** d’un graphe orienté ou non orienté  
- **Affichage** de la liste d’adjacence  
- **Visualisation** graphique (via `networkx` + `matplotlib`)  
- **Densité** du graphe  
- **Degré** (maximum) du graphe  
- **Test eulérien** (eulerian)  
- **Test de complétude**  
- **Sous‐graphe complet maximal**  
- **Tous les chemins** entre deux nœuds  
- **Plus court chemin** entre deux nœuds (via recherche exhaustive)  
- **Composantes k-connexes**  
- **Détection de cycles**  
- **Cycle Hamiltonien**  
- **Test de k-clique**  
- **Clique maximale**  

---

## 📦 Prérequis

- Python 3.7+  
- (Optionnel pour la visualisation)

  ```bash
  pip install networkx matplotlib
## Exemple rapide :
```
  ==============================
  Programme gestion des graphes:
  ==============================
  (1) Construction d'un graphe orienté/non orienté
  (2) Affichage du graphe
  (3) Calculer la densité du graphe
  (4) Calculer le degré du graphe
  (5) Vérifier si le graphe est eulérien
  (6) Vérifier si le graphe est complet
  (7) Trouver un sous-graphe complet maximal
  (8) Recherche de tous les chemins entre un nœud a et un nœud b
  (9) Recherche du chemin le plus court entre deux nœuds a et b
  (10) Recherche de composantes (fortement) connexes
  (11) Trouver tous les cycles/circuits dans le graph
  (12) Vérifier si le graphe contient un cycle/circuit hamiltonien
  (13) Vérifier si le graphe contient une k-clique
  (14) Trouver une crique maximale dans un graphe G
  Entrer un nombre entre 1 et 14: 1
  Choisir le type du graphe :
  * Oriente :1
  * Non oriente: 0
  > 0
  Construisez votre graphe non orienté :
  Entrez le nombre de sommets : 3
  - Sommet 1: A
  - Sommet 2: B
  - Sommet 3: C
  Entrez le nombre d'arêtes : 2
  - Arête : A B
  - Arête : B C
