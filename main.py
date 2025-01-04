#!/usr/bin/python3
print("==============================")
print("Programme gestion des graphes:")
print("==============================")
print("(1) Construction d'un graphe orienté/non orienté")
print("(2) Affichage du graphe")
print("(3) Calculer la densité du graphe")#!
print("(4) Calculer le degré du graphe")#!
print("(5) Vérifier si le graphe est eulérien")#!
print("(6) Vérifier si le graphe est complet")#!
print("(7) Trouver un sous-graphe complet maximal")#!
print("(8) Recherche de tous les chemins entre un nœud a et un nœud b")#!
print("(9) Recherche du chemin le plus court entre deux nœuds a et b")#!
print("(10) Recherche de composantes (fortement) connexes à partir d'un nœud a.")#!
print("(11) Recherche des composantes 4-connexes dans le graphe")#!
print("(12) Trouver tous les cycles/circuits dans le graph")#!

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
        
        # Calcul du nombre d'arêtes
        for voisins in self.listeAdj.values():
            num_aretes = num_aretes + len(voisins) # calculer la somme des aretes
        num_aretes = num_aretes // 2 # diviser le nombre d'arete sur 2
        
        # Calcul de la densité
        max_possible_aretes = ordre_graphe * (ordre_graphe - 1)
        max_possible_aretes = max_possible_aretes // 2  # Graphe non orienté
        
        densite = num_aretes / max_possible_aretes
        return densite

    def degreeGraphe(self):
        degree = 0
        for i in self.listeAdj.values():
            num_aretes = len(i)
            if(degree < num_aretes):
                degree = num_aretes
        return degree

    def grapheEulerien(self):

        def is_connected(graph):
            """Vérifie si un graphe non orienté est connexe."""
            visited = set()
            nodes = list(graph.keys())
            
            def dfs(node):
                if node not in visited:
                    visited.add(node)
                    for neighbor in graph[node]:
                        dfs(neighbor)
            
            dfs(nodes[0])
            return len(visited) == len(nodes)

        def is_eulerian(graph):
            """Vérifie si un graphe non orienté est eulérien."""
            # Vérifier la connectivité
            if not is_connected(graph):
                return False
            
            # Vérifier si tous les sommets ont un degré pair
            for node, neighbors in graph.items():
                if len(neighbors) % 2 != 0:
                    return False
            
            return True

        if is_eulerian(self.listeAdj):
            print("Le graphe est eulérien.")
            return True
        else:
            print("Le graphe n'est pas eulérien.")
            return False

    def estComplet(self):
        """
        Vérifie si un graphe non orienté est complet.
        
        :param graph: dict, représentation du graphe sous forme d'adjacence {sommet: {voisins}}
        :return: bool, True si le graphe est complet, False sinon
        """
        graph=self.listeAdj
        num_nodes = len(graph)  # Nombre de sommets
        
        # Vérifier les connexions de chaque sommet
        for node, neighbors in graph.items():
            # Dans un graphe complet, chaque sommet doit être connecté à tous les autres sommets
            if len(neighbors) != num_nodes - 1:
                return False
            
            # Vérifier que tous les voisins sont dans les sommets du graphe
            for neighbor in neighbors:
                if neighbor not in graph:
                    return False
        
        return True

    def sousGrapheComplet(self):
        graph=self.listeAdj
        from itertools import combinations
        def is_clique(graph, nodes):
            """
            Vérifie si un ensemble de sommets 'nodes' forme une clique.
            
            :param graph: dict, représentation du graphe sous forme de liste d'adjacence
            :param nodes: list, ensemble de sommets à tester
            :return: bool, True si c'est une clique, False sinon
            """
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if nodes[j] not in graph[nodes[i]]:
                        return False
            return True
        """
        Trouve le plus grand sous-graphe complet (clique maximale).
        
        :param graph: dict, représentation du graphe sous forme de liste d'adjacence
        :return: list, sommets de la clique maximale
        """
        nodes = list(graph.keys())
        max_clique = []
        
        # Tester toutes les combinaisons de sommets, de taille décroissante
        for size in range(len(nodes), 0, -1):
            for combination in combinations(nodes, size):
                if is_clique(graph, combination):
                    return list(combination)  # Retourner la première clique maximale trouvée
        
        return max_clique

    def find_all_paths(self, start, end, path=None):
        graph=self.listeAdj
        """
        Trouve tous les chemins entre deux nœuds dans un graphe.
        
        :param graph: dict, représentation du graphe sous forme de liste d'adjacence
        :param start: str, nœud de départ
        :param end: str, nœud de destination
        :param path: list, chemin actuel (utilisé dans la récursion)
        :return: list, liste de tous les chemins
        """
        if path is None:
            path = []
        
        path = path + [start]  # Ajouter le nœud actuel au chemin
        
        # Si le nœud de départ est le nœud de destination
        if start == end:
            return [path]
        
        # Si le nœud de départ n'a pas de voisins
        if start not in graph:
            return []
        
        paths = []
        for neighbor in graph[start]:
            if neighbor not in path:  # Éviter les cycles
                new_paths = self.find_all_paths(neighbor, end, path)
                for new_path in new_paths:
                    paths.append(new_path)
        
        return paths

    def cheminPlusCourt(self,start,end):
        return min(self.find_all_paths(start,end),key=len)

    def find_cycles(self):
        """
        Trouve tous les cycles dans un graphe non orienté.
        
        :param graph: dict, représentation du graphe sous forme de liste d'adjacence
        :return: list, liste contenant tous les cycles trouvés
        """
        graph=self.listeAdj
        def dfs(node, parent, visited, path):
            """
            Fonction récursive pour effectuer une recherche DFS et détecter les cycles.
            
            :param node: noeud actuel
            :param parent: noeud parent (évite les cycles bidirectionnels dans les graphes non orientés)
            :param visited: ensemble des nœuds visités
            :param path: liste contenant le chemin actuel
            """
            visited.add(node)
            path.append(node)
            
            for neighbor in graph[node]:
                # Si le voisin n'a pas été visité, continuer la recherche DFS
                if neighbor not in visited:
                    dfs(neighbor, node, visited, path)
                # Si un cycle est trouvé (le voisin est visité mais n'est pas le parent)
                elif neighbor != parent:
                    cycle = path[path.index(neighbor):]  # Extraire le cycle
                    cycles.append(cycle)
            
            # Retour en arrière
            path.pop()
        
        visited = set()
        cycles = []
        
        for node in graph:
            if node not in visited:
                dfs(node, None, visited, [])
        
        # Éliminer les doublons (par exemple, [A, B, C] est le même que [B, C, A])
        unique_cycles = []
        for cycle in cycles:
            cycle_set = set(cycle)
            if cycle_set not in unique_cycles:
                unique_cycles.append(cycle_set)
        
        return [list(cycle) for cycle in unique_cycles]

    def find_k_connected_components(self, k):
        """
        Trouve les composantes k-connexes dans un graphe non orienté.
        
        :param graph: dict, représentation du graphe sous forme de liste d'adjacence
        :param k: int, degré de connexion minimum requis
        :return: list, liste des composantes k-connexes
        """
        graph=self.listeAdj
        def dfs(node, visited):
            """
            Effectue une recherche DFS pour identifier une composante connexe.
            
            :param node: noeud actuel
            :param visited: ensemble des nœuds déjà visités
            :return: list, la composante connexe trouvée
            """
            stack = [node]
            component = []
            
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    stack.extend(graph[current] - visited)
            
            return component

        # Étape 1 : Trouver toutes les composantes connexes
        visited = set()
        components = []
        for node in graph:
            if node not in visited:
                component = dfs(node, visited)
                components.append(component)

        # Étape 2 : Filtrer les composantes k-connexes
        k_connected_components = []
        for component in components:
            # Construire le sous-graphe de la composante
            subgraph = {n: graph[n] & set(component) for n in component}
            
            # Vérifier si chaque nœud a au moins k connexions
            if all(len(neighbors) >= k for neighbors in subgraph.values()):
                k_connected_components.append(component)

        return k_connected_components



#! class Graph_oriente:
#    def __init__(self):
#         self.sommets={}
#     def ajoutSommet(self,s):
#         if s not in self.sommets.keys():
#             self.sommets[s]={'suc':set(),'pred':set()}


#! Saisie du choix d'algorithme
# print("Entrer votre choix:")
# choix = None
# while choix is None:
#     try:
#         choix = int(input(""))
#         if choix<1 or choix >12:
#             choix=None
#             raise ValueError
#     except ValueError:
#         print("Nombre invalid!")

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




# Exemple d'utilisation
graph1,graph2=Graph(False),Graph(False)

graph1.listeAdj = {
    'A': {'B', 'C', 'D'},
    'B': {'A', 'C', 'D'},
    'C': {'A', 'B', 'D'},
    'D': {'A', 'B', 'C'}
}

graph2.listeAdj = {
    'A': {'B', 'C'},
    'B': {'A', 'C'},
    'C': {'A', 'B'},
    'D': {'C'}
}

graph=Graph(False)

graph.listeAdj = {
    'A': {'B', 'C'},
    'B': {'A','C','D'},
    'C': {'A','C'},
    'D': {'B', 'E'},
    'E': {'D'}
}


# print("Graphe 1 est complet :", graph1.estComplet())  # True
# print("Graphe 2 est complet :", graph2.estComplet())  # False


# Exemple d'utilisation
g=Graph(False)
g.listeAdj = {
    'A': {'B', 'C'},
    'B': {'A', 'D', 'E'},
    'C': {'A', 'F'},
    'D': {'B'},
    'E': {'B', 'F'},
    'F': {'C', 'E'}
}

# start_node = 'A'
# end_node = 'C'

# paths = graph2.find_all_paths(start_node, end_node)
# print(f"Tous les chemins de {start_node} à {end_node} :")
# for path in paths:
#     print(" -> ".join(path))

# print(g.cheminPlusCourt('A','F'))


# Exemple d'utilisation
graph3=Graph(False)
graph3.listeAdj = {
    'A': {'B', 'C'},
    'B': {'A', 'C', 'D'},
    'C': {'A', 'B', 'E'},
    'D': {'B', 'E'},
    'E': {'C', 'D'}
}

# cycles = graph3.find_cycles()
# print("Cycles trouvés :")
# for cycle in cycles:
#     print(" -> ".join(cycle))


# Exemple d'utilisation
graph4=Graph(False)
graph4.listeAdj = {
    'A': {'B', 'C', 'D', 'E'},
    'B': {'A', 'C', 'D'},
    'C': {'A', 'B', 'D'},
    'D': {'A', 'B', 'C', 'E'},
    'E': {'A', 'D'}
}

# k = 4  # Niveau de connexité recherché
# k_connected = graph4.find_k_connected_components(k)

# print(f"Composantes {k}-connexes :")
# for component in k_connected:
#     print(component)

# graph3.afficherGraphe()
