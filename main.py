#!/usr/bin/python3
def printApp():
    print("==============================")
    print("Programme gestion des graphes:")
    print("==============================")
    print("(1) Construction d'un graphe orienté/non orienté")
    print("(2) Affichage du graphe")
    print("(3) Calculer la densité du graphe")
    print("(4) Calculer le degré du graphe")
    print("(5) Vérifier si le graphe est eulérien")
    print("(6) Vérifier si le graphe est complet")
    print("(7) Trouver un sous-graphe complet maximal")
    print("(8) Recherche de tous les chemins entre un nœud a et un nœud b")
    print("(9) Recherche du chemin le plus court entre deux nœuds a et b")
    print("(10) Recherche de composantes (fortement) connexes à partir d'un nœud a.")
    print("(11) Trouver tous les cycles/circuits dans le graph")
    print("(12) Vérifier si le graphe contient un cycle/circuit hamiltonien")
    print("(13) Vérifier si le graphe contient une k-clique (k donné)")
    print("(14) Trouver une crique maximale dans un graphe G")

class Graph:
    def __init__(self, oriente = False):
        self.listeAdj = {}
        self.oriente = oriente

    def ajoutSommet(self,s):
        if s not in self.listeAdj.keys():
            if self.oriente == False:
                self.listeAdj[s]=set() # creer un sommet et garder l'unicite de chaque element
            else:
                self.listeAdj[s]={'succ':set(),'pred':set()}

    def ajoutLien(self,p,s):
        if p and s in self.listeAdj.keys():
            if self.oriente == False:
                self.listeAdj[p].add(s)
                self.listeAdj[s].add(p)
            else:
                self.listeAdj[p]['succ'].add(s)
                self.listeAdj[s]['pred'].add(p)

    def afficherListeAdj(self):
        """
        Affiche la représentation mémoire d'un graphe sous forme de liste d'adjacence.
        
        :param graph: dict, représentation du graphe
        """
        print("Représentation du Graphe :")
        if self.oriente == False:
            for node, neighbors in self.listeAdj.items():
                print(f"{node} -> {', '.join(neighbors)}")
        else:
            for node, neighbors in self.listeAdj.items():
                print(f"{node} -> {', '.join(neighbors['succ'])}") 

    def afficherGraphe(self):
        # Importer les bibliothèques pour l'affichage
        import networkx as nx # type: ignore
        import matplotlib.pyplot as plt # type: ignore

        # Convertir la structure en un graphe NetworkX
        graph = nx.Graph()  # Utilise nx.DiGraph() si le graphe est orienté


        # Add nodes
        for node in self.listeAdj:
            graph.add_node(node)  # Add all nodes

        # Add edges (if they exist)
        for node, neighbors in self.listeAdj.items():
            for neighbor in neighbors:
                graph.add_edge(node, neighbor)

        # Dessiner et afficher le graphe
        plt.figure(figsize=(8, 6))
        # Handle empty graphs (with no edges) by forcing node positions
        if len(graph.edges) == 0:
            pos = nx.circular_layout(graph)  # Spread out nodes in a circular layout
        else:
            pos = nx.spring_layout(graph)  # Default spring layout for connected graphs

        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_color='skyblue',
            font_weight='bold',
            node_size=2000
        )
        plt.title("Graphe avec des Voisins sous Forme d'Ensembles")
        plt.show()

    def densiteGraphe(self):

        ordre_graphe = len(self.listeAdj.keys()) # calculer le degree du graphe avec la fonction longueur
        if ordre_graphe <= 1:
            # Si un seul nœud ou aucun, densité = 0
            return 0
        
        # Calcul de la densité
        max_possible_aretes = ordre_graphe * (ordre_graphe - 1)
        
        # Calcul du nombre d'arêtes
        num_aretes = 0
        for voisins in self.listeAdj.values():
            if(self.oriente == False):
                num_aretes = num_aretes + len(voisins) # calculer la somme des aretes
            else:
                num_aretes = num_aretes + len(voisins['succ'])
        if(self.oriente == False):
            num_aretes = num_aretes // 2 # diviser le nombre d'arete sur 2
            max_possible_aretes = max_possible_aretes // 2  # Graphe non orienté

        densite = num_aretes / max_possible_aretes
        return densite

    def degreeGraphe(self):
        degree = 0 # initialise le degree a 0
        for i in self.listeAdj.values(): # boucler sur les liste des voisins
            if self.oriente == False:
                num_aretes = len(i) # calculer le nombre d'aretes
            else:
                num_aretes = len(i['succ']) + len(i['pred']) # nombre d'arcs pour graphe non oriente
            if(degree < num_aretes):
                degree = num_aretes # determiner si le degree de cette sommet est le degre du graphe
        return degree

    def est_eulerian(self):
        graph = self.listeAdj
        def est_connecte_oriente(graph):
            """
            Vérifie si un graphe orienté est fortement connexe.
            """
            def dfs(node, visite, direction):
                visite.add(node)
                for neighbor in graph[node][direction]:
                    if neighbor not in visite:
                        dfs(neighbor, visite, direction)

            nodes = list(graph.keys())
            # Vérification dans la direction 'succ'
            visite = set()
            dfs(nodes[0], visite, 'succ')
            if len(visite) != len(graph):
                return False

            # Vérification dans la direction 'pred'
            visite = set()
            dfs(nodes[0], visite, 'pred')
            if len(visite) != len(graph):
                return False

            return True

        def est_connecte(graph):
            """
            Vérifie si un graphe non orienté est connexe.
            """
            visite = set()

            def dfs(node):
                visite.add(node)
                for voisin in graph[node]:
                    if voisin not in visite:
                        dfs(voisin)

            nodes = list(graph.keys())
            dfs(nodes[0])
            return len(visite) == len(graph)

        # Pour les graphes orientés
        if self.oriente == True:
            # Vérification de la forte connexité
            if not est_connecte_oriente(graph): # verifier si le graphe est fortement connecte
                return False

            # Vérification des degrés entrants = degrés sortants
            for node, aretes in graph.items():
                if len(aretes['succ']) != len(aretes['pred']): # verifier si chaque noeud a un degree pair
                    return False

            return True

        # Pour les graphes non orientés
        else:
            # Vérification de la connexité
            if not est_connecte(graph): # verifier si le graphe est fortement connecte
                return False

            # Vérification des degrés pairs
            for node, voisins in graph.items():
                if len(voisins) % 2 != 0: # verifier si chaque noeud a un degree pair
                    return False

            return True

    def estComplet(self):
        graph = self.listeAdj
        num_nodes = len(graph.keys())  # Nombre de sommets
        
        # Un graphe avec 0 ou 1 sommet est complet par définition
        if num_nodes <= 1:
            return True

        if self.oriente == True:
            # Vérifier les connexions de chaque sommet
            for node, voisins in graph.items():
                # Dans un graphe complet, chaque sommet doit être connecté à tous les autres sommets
                if len(voisins['succ']) != num_nodes - 1 or len(voisins['pred']) != num_nodes - 1:
                    return False
        else:
            # Vérifier les connexions de chaque sommet
            for node, voisins in graph.items():
                # Dans un graphe complet, chaque sommet doit être connecté à tous les autres sommets
                if len(voisins) != num_nodes - 1:
                    return False
        return True

    def sousGrapheComplet(self):
        graph = self.listeAdj
        from itertools import combinations
        def is_clique(graph, nodes):
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    # Vérifier pour un graphe orienté
                    if self.oriente == True:
                        if nodes[j] not in graph[nodes[i]]['succ'] or \
                            nodes[i] not in graph[nodes[j]]['pred']:
                            return False
                    else:
                        # Vérifier pour un graphe non orienté
                        if nodes[j] not in graph[nodes[i]]:
                            return False
            return True

        nodes = list(graph.keys())
        max_clique = []
        
        # Tester toutes les combinaisons de sommets, de taille décroissante
        for size in range(len(nodes), 0, -1):
            for combination in combinations(nodes, size):
                if is_clique(graph, combination):
                    return list(combination)  # Retourner la première clique maximale trouvée
        
        return max_clique

    def find_all_paths(self, debut, fin, path=None):
        graph = self.listeAdj
        if path is None:
            path = []
        
        path = path + [debut]  # Ajouter le nœud actuel au chemin
        
        # Si le nœud de départ est le nœud de destination
        if debut == fin:
            return [path]
        
        # Si le nœud de départ n'a pas de voisins
        if debut not in graph:
            return []
        
        paths = []

        if self.oriente == True:
            voisins = graph[debut]['succ']
        else:
            voisins = graph[debut]

        for voisin in voisins:
            if voisin not in path:  # Éviter les cycles
                new_paths = self.find_all_paths(voisin, fin, path)
                for new_path in new_paths:
                    paths.append(new_path)
        
        return paths

    def cheminPlusCourt(self,debut,fin):
        return min(self.find_all_paths(debut,fin),key=len)

    def find_cycles(self):
        """
        Trouve tous les cycles dans un graphe (orienté ou non orienté).
        :return: Liste des cycles trouvés.
        """
        graph = self.listeAdj

        def dfs(node, parent, visite, path):
            """
            Recherche DFS pour détecter les cycles.
            """
            visite.add(node)
            path.append(node)
            
            if self.oriente == True:  # Graphe orienté
                voisins = graph[node].get('succ', set())
            else:  # Graphe non orienté
                voisins = graph[node]

            for voisin in voisins:
                if voisin not in visite:  # Continuer la recherche DFS
                    dfs(voisin, node, visite, path)
                elif voisin != parent:  # Cycle détecté
                    # Extraire le cycle
                    cycle_start_index = path.index(voisin)
                    cycle = path[cycle_start_index:]
                    # Stocker le cycle sous forme canonique (trié)
                    cycle_sorted = tuple(sorted(cycle))
                    if cycle_sorted not in unique_cycles:
                        unique_cycles.add(cycle_sorted)

            # Retour en arrière
            path.pop()
        
        visite = set()
        unique_cycles = set()  # Utilisé pour éliminer les doublons
        
        for node in graph:
            if node not in visite:
                dfs(node, None, visite, [])
        
        # Retourner les cycles sous forme de liste de listes
        return [list(cycle) for cycle in unique_cycles]
    
    def composantes_k_connexes(self, k=4):
        def dfs_k_connexes(node, visited, component):
            """
            Perform DFS to find k-connected components.
            """
            visited.add(node)
            component.add(node)
            if self.oriente:
                neighbors = self.listeAdj[node].get('succ', set())
            else:
                neighbors = self.listeAdj[node]

            for neighbor in neighbors:
                if neighbor not in visited and len(neighbors) >= k:
                    dfs_k_connexes(neighbor, visited, component)

        visited = set()
        components = []
        for node in self.listeAdj:
            if node not in visited and len(self.listeAdj[node]) >= k:
                component = set()
                dfs_k_connexes(node, visited, component)
                components.append(component)

        return components

    def hamiltonian_cycle(self):
        def backtrack(node, visited, path):
            path.append(node)
            visited.add(node)

            if len(visited) == len(self.listeAdj):  # All nodes visited
                if path[0] in (self.listeAdj[node] if not self.oriente else self.listeAdj[node].get('succ', set())):
                    return path + [path[0]]  # Return to start node
                visited.remove(node)
                path.pop()
                return None

            neighbors = self.listeAdj[node] if not self.oriente else self.listeAdj[node].get('succ', set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    result = backtrack(neighbor, visited, path)
                    if result:
                        return result

            visited.remove(node)
            path.pop()
            return None

        for start_node in self.listeAdj:
            result = backtrack(start_node, set(), [])
            if result:
                return result
        return None

    def has_k_clique(self, k):
        from itertools import combinations
        nodes = list(self.listeAdj.keys())

        for subset in combinations(nodes, k):
            # Check for k-clique in directed or undirected graph
            if all(
                (node2 in self.listeAdj[node1] if not self.oriente else node2 in self.listeAdj[node1].get('succ', set()))
                and (node1 in self.listeAdj[node2] if not self.oriente else node1 in self.listeAdj[node2].get('succ', set()))
                for node1 in subset for node2 in subset if node1 != node2
            ):
                return True
        return False

    def maximal_clique(self):
        def is_clique(nodes):
            return all(
                (node2 in self.listeAdj[node1] if not self.oriente else node2 in self.listeAdj[node1].get('succ', set()))
                and (node1 in self.listeAdj[node2] if not self.oriente else node1 in self.listeAdj[node2].get('succ', set()))
                for node1 in nodes for node2 in nodes if node1 != node2
            )

        def backtrack(curr_clique, candidates):
            if not candidates:
                return curr_clique

            max_clique = curr_clique[:]
            for node in candidates:
                new_clique = curr_clique + [node]
                if is_clique(new_clique):
                    remaining_candidates = [n for n in candidates if n != node]
                    candidate_clique = backtrack(new_clique, remaining_candidates)
                    if len(candidate_clique) > len(max_clique):
                        max_clique = candidate_clique

            return max_clique

        nodes = list(self.listeAdj.keys())
        return backtrack([], nodes)


def constGraph():
    """
    Fonction pour construire un graphe à partir de l'entrée utilisateur.
    Retourne un graphe sous forme de dictionnaire d'adjacence.
    """
    type = input("Choisir le type du graphe :\n* Oriente :1\n* Non oriente: 0\n")
    while type != '1' and type != '0':
        type = input("choix non valide entrer 1 ou 0 svp :")
    if type == '1':
        print("Construisez votre graphe orienté :")
        oriente = True
    else:
        print("Construisez votre graphe non orienté :")
        oriente = False


    graph = Graph(oriente)

    # Nombre de sommets
    num_nodes = int(input("Entrez le nombre de sommets : "))

    for i in range(num_nodes):
        node = input(f"- Sommet {i+1}: ").strip()
        graph.ajoutSommet(node)
    
    # Nombre d'arêtes
    num_edges = int(input("Entrez le nombre d'arêtes : "))
    
    print("Entrez les arêtes sous forme de paires (ex: A B) :")
    for _ in range(num_edges):
        u, v = input("- Arête : ").strip().split()
        if u in graph.listeAdj.keys() and v in graph.listeAdj.keys():
            graph.ajoutLien(u,v)
        else:
            print(f"Erreur : {u} ou {v} n'existe pas. Réessayez.")
    return graph



app_graphe = Graph(False)

import os
# ! Saisie du choix d'algorithme
print("Entrer votre choix:")
choix = None
while choix is None:
    try:
        printApp()
        choix = int(input("Entrer un nombre entre 1 et 12:"))
        if choix<1 or choix >12:
            choix=None
            raise ValueError
        else:
            match choix:
                case 1:
                    app_graphe = constGraph()

                case 2:
                    app_graphe.afficherListeAdj()
                    if app_graphe.oriente == False:
                        app_graphe.afficherGraphe()

                case 3:
                    print(app_graphe.densiteGraphe())

                case 4:
                    print(app_graphe.degreeGraphe())

                case 5:
                    print(app_graphe.est_eulerian())

                case 6:
                    print(app_graphe.estComplet())

                case 7:
                    print(app_graphe.sousGrapheComplet())

                case 8:
                    deb = input("Entrer le 1er noeud:")
                    fin = input("Entrer le 2eme noeud:")
                    paths = app_graphe.find_all_paths(deb,fin)
                    if len(paths)<1:
                        print("pas de chemin entre ces deux sommets")
                    else:
                        print(paths)

                case 9:
                    deb = input("Entrer le 1er noeud:")
                    fin = input("Entrer le 2eme noeud:")
                    paths = app_graphe.cheminPlusCourt(deb,fin)
                    if len(paths)<1:
                        print("pas de chemin entre ces deux sommets")
                    else:
                        print(paths)

                case 10:
                    k = int(input("Donner k(k=4 si k non fournis):"))
                    if type(k) == int:
                        print(app_graphe.composantes_k_connexes(k))
                    else:
                        print(app_graphe.composantes_k_connexes())


                case 11:
                    print(app_graphe.find_cycles())

                case 12:
                    print(app_graphe.hamiltonian_cycle())

                case 13:
                    print(app_graphe.has_k_clique())

                case 14:
                    print(app_graphe.maximal_clique())
        choix = None
        os.system("clear")
    except ValueError:
        print("Nombre invalid!")

