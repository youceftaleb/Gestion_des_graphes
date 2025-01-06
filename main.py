#!/usr/bin/python3
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
        
        # Calcul de la densité
        max_possible_aretes = ordre_graphe * (ordre_graphe - 1)
        
        # Calcul du nombre d'arêtes
        num_aretes = 0
        for voisins in self.listeAdj.values():
            if(self.oriente == False):
                num_aretes = num_aretes + len(voisins) # calculer la somme des aretes
                num_aretes = num_aretes // 2 # diviser le nombre d'arete sur 2
                max_possible_aretes = max_possible_aretes // 2  # Graphe non orienté
            else:
                num_aretes = num_aretes + len(voisins['succ'])

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
        graph=self.listeAdj
        num_nodes = len(graph.keys())  # Nombre de sommets
        
        # Un graphe avec 0 ou 1 sommet est complet par définition
        if num_nodes <= 1:
            return True

        if self.oriente == True:
            # Vérifier les connexions de chaque sommet
            for node, voisins in graph.items():
                # Dans un graphe complet, chaque sommet doit être connecté à tous les autres sommets
                if len(voisins['succ']) != num_nodes - 1 or len(voisins['pred'] != num_nodes):
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
        graph=self.listeAdj
        def dfs(node, parent, visited, path):

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
