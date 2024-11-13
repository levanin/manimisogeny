import networkx as nx
# A new script to generate SS isogeny graphs
# Because I couldn't find existing methods that correctly keep track of edge multiplicity and cycles nicely.
# Author: Shai Levin (2024)

def supersingular_ell_isogeny_graph(l, p):
    # Returns the full supersingular l-isogeny graph over characteristic p
    if (p % 4 != 3 and p % 3 != 2) or not is_prime(l) or not is_prime(p):
        raise ValueError("l,p must be prime where p = 3 (mod 4)")
    F = GF(p^2)
    Fp4 = F.extension(2)
    vertex_set = []
    edge_set = []
    if p % 3 == 2:
        vertex_set.append(F(0))
    if p % 4 == 3:
        vertex_set.append(F(1728))
    frontier = vertex_set.copy()
    while len(frontier) > 0:
        frontier, vertex_set, edge_set = update(l,frontier, vertex_set, edge_set)
    return Graph([vertex_set, edge_set], multiedges=True, loops=True)

def update(l, frontier, vertex_set, edge_set):
    # For each unexplored vertex in the frontier, it computes the l-isogenous neighbours
    # If the neighbour is not in the vertex_set, it adds the neighbour to the frontier, vertex_set and the edges to the edge_set
    # If the neighbour is already in the vertex_set, but there is no edge between the frontier vertex and the neighbour, it adds the edge to the edge_set
    # If the neighbour is already in the vertex_set, and the edge is in the edge_set, does nothing
    new_frontier = []
    new_vertex_set = vertex_set
    new_edge_set = edge_set
    for j in frontier:
        for neighbour, multiplicity in neighbours(j, l, extend=True):
            if neighbour not in new_vertex_set:
                new_frontier.append(neighbour)
                new_vertex_set.append(neighbour)
                for _ in range(multiplicity):
                    new_edge_set.append((j, neighbour))
            elif (j, neighbour) not in new_edge_set and (neighbour, j) not in new_edge_set:
                for _ in range(multiplicity):
                    new_edge_set.append((j, neighbour))
    return new_frontier, new_vertex_set, new_edge_set

def neighbours(j, l, extend=False):
    # Returns the l-isogenous neighbours of j over F
    # If extend is True, it returns the roots of the modular polynomial over the extension of F (since the polynomial may only be reducible over the extension)
    if extend==True:
        F = j.parent()
        Fp4 = F.extension(2)
        return [(F(root), multiplicity) for (root, multiplicity) in classical_modular_polynomial(l, Fp4(j)).roots() if root in F]
    return classical_modular_polynomial(l, j).roots()

# Some nice primes

p = 3*2^7 - 1

G = supersingular_ell_isogeny_graph(2, p)

print([[str(x),str(y)] for (x,y,z) in G.edges()])

H = G.networkx_graph()
nx.write_adjlist(H, "adjlist.txt", delimiter=',')
adj_list = nx.read_adjlist("adjlist.txt", delimiter=',')
print(list(adj_list))