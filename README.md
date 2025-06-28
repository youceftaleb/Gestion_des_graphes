# 🧠 Programme de Gestion de Graphes

Un outil en ligne de commande pour **créer**, **analyser** et **visualiser** des **graphes orientés ou non orientés**.

Ce projet a été réalisé dans le cadre du module **Algorithmique Avancée et Complexité** (AAC), en Master 1 Bio-Informatique à l’Université, et est accompagné d’un rapport détaillé (`AAC_Projet.pdf`) expliquant les fondements théoriques, les algorithmes utilisés et leurs applications, notamment en bio-informatique.

---

## ✨ Fonctionnalités

- ✅ Construction d’un graphe (orienté ou non orienté)  
- 📋 Affichage de la **liste d’adjacence**  
- 📊 **Visualisation graphique** (`networkx` + `matplotlib`)  
- 📈 Calcul de la **densité** du graphe  
- 📏 Calcul du **degré maximal**  
- 🔁 **Test eulérien**  
- 🔎 **Vérification de complétude**  
- 🧩 Détection d’un **sous-graphe complet maximal**  
- 🧭 Recherche de **tous les chemins** entre deux sommets  
- 🚀 **Plus court chemin** (recherche exhaustive)  
- 🧱 Détection des **composantes connexes / k-connexes**  
- 🔄 Détection de **cycles**  
- 🧮 Vérification de **cycle Hamiltonien**  
- 📌 Test de **k-clique**  
- 🧠 Recherche de **clique maximale**

---

## 📄 Rapport

Le fichier [`AAC_Projet.pdf`](AAC_Projet.pdf) contient :

- Une **introduction complète à la théorie des graphes**
- Des **définitions formelles** (graphes simples, orientés, bipartis, etc.)
- Une **explication des représentations mémoire** (matrice/liste d’adjacence)
- L’**implémentation d’algorithmes** avec leur **complexité**
- Une **étude expérimentale** des performances (temps d'exécution)
- Une **analyse des applications en bio-informatique** (génomique, réseaux PPI, etc.)

---

## 📦 Prérequis

- Python 3.7+
- Pour la visualisation (optionnelle) :

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
