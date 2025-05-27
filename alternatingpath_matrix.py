from clausesets import ClauseSet
from alternatingpath_abstract import RelevanceGraph
from fofspec import FOFSpec
from literals import Literal
from clauses import Clause
from unification import mgu
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import dijkstra

class MatrixRelevanceGraph(RelevanceGraph):

    def __init__(self, clause_set):
        self.construct_graph(clause_set)
        return

    def construct_graph(self, clause_set):
        self.nodes = self.construct_nodes(clause_set)
        self.adjacency_matrix = self.construct_matrix()

    def clauses_to_nodes(self, clauses):
        return np.nonzero(np.isin(self.nodes[:,0], clauses.clauses))[0]

    def construct_nodes(self, clauses):
        nodes = []
        for clause in clauses.clauses:
            for literal in clause.literals:
                for _ in range(2):
                    nodes.append((clause, literal, literal.negative, literal.atom))
        return np.array(nodes, dtype=object)

    def construct_matrix(self):
        print("shape ", 2*[self.nodes.shape[0]])
        return np.fromfunction(
            self.is_connected,
            shape=2*[self.nodes.shape[0]],
            dtype=int
        )

    def is_connected(self, i, j):
        nodes_i = self.nodes[i,:]
        nodes_j = self.nodes[j,:]

        different_direction = (np.indices(i.shape).sum(axis=0) % 2).astype(bool)

        same_clause = nodes_i[:,:,0] == nodes_j[:,:,0]
        different_literal = nodes_i[:,:,1] != nodes_j[:,:,1]

        different_literal_sign = nodes_i[:,:,2] != nodes_j[:,:,2]
        mgu_exists_vectorized = np.vectorize(lambda x, y: mgu(x,y) is not None)
        mgu_exists = mgu_exists_vectorized(nodes_i[:,:,1],nodes_j[:,:,1])

        in_clause_connected = same_clause & different_literal
        between_clause_connected = ~same_clause & different_literal_sign & mgu_exists
        return different_direction & (in_clause_connected | between_clause_connected)



    def get_rel_neighbourhood(self, from_clauses, distance):

        from_nodes = self.clauses_to_nodes(from_clauses)

        distances_per_node = dijkstra(
            csgraph=self.adjacency_matrix,
            indices=from_nodes,
            directed=False,
            unweighted=True,
        )

        distance_to_set = np.min(distances_per_node, axis=0)
        rel_neighbourhood = distance_to_set <= (2 * distance - 1)

        print(rel_neighbourhood)

        return

