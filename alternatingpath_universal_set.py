from clauses import Clause
from clausesets import ClauseSet
from literals import Literal
from unification import mgu
from collections import defaultdict
from alternatingpath_abstract import RelevanceGraph


class Node:
    def __init__(self, literal: Literal, clause: Clause) -> None:
        self.literal: Literal = literal
        self.clause: Clause = clause

    def __repr__(self) -> str:
        return f"<{self.clause.name},{self.literal}>"


class Edge:
    def __init__(self, node1: Node, node2: Node) -> None:
        self.node1: Node = node1
        self.node2: Node = node2

    def __repr__(self) -> str:
        return f"Edge: {self.node1} - {self.node2}"


class UniversalSetRelevanceGraph(RelevanceGraph):

    def construct_graph(self, clause_set: ClauseSet) -> None:
        self.out_nodes, self.in_nodes = self.construct_nodes(clause_set)
        self.edges: set[Edge] = (
            self.construct_inclause_edges() | self.construct_betweenclause_edges()
        )

    @staticmethod
    def construct_nodes(clause_set: ClauseSet):
        out_nodes = set()
        in_nodes = set()
        for clause in clause_set.clauses:
            for literal in clause.literals:
                out_nodes.add(Node(literal, clause))
                in_nodes.add(Node(literal, clause))
        return out_nodes, in_nodes

    def construct_inclause_edges(self):
        in_clause_edges = set()
        for in_node in self.in_nodes:
            for out_node in self.out_nodes:
                if in_node.clause != out_node.clause:
                    continue
                if in_node.literal == out_node.literal:
                    continue
                in_clause_edges.add(Edge(in_node, out_node))
        return in_clause_edges

    def construct_betweenclause_edges(self):
        between_clause_edges = set()
        for out_node in self.out_nodes:
            for in_node in self.in_nodes:
                if out_node.literal.negative == in_node.literal.negative:
                    continue
                if mgu(out_node.literal.atom, in_node.literal.atom) == None:
                    continue
                between_clause_edges.add(Edge(out_node, in_node))
        return between_clause_edges

    def get_all_nodes(self):
        return self.out_nodes | self.in_nodes

    @staticmethod
    def nodes_to_clauses(nodes):
        return ClauseSet({node.clause for node in nodes})

    def clauses_to_nodes(self, clauses: ClauseSet):
        nodesOfClauseSubset = {
            node for node in self.out_nodes if node.clause in clauses.clauses
        }
        return nodesOfClauseSubset

    @staticmethod
    def edge_neighb_of_subset(edge: Edge, subset: set[Node]):
        return (edge.node1 in subset) != (edge.node2 in subset)

    def get_neighbours(self, subset: set[Node]):
        neighbouring_edges = {
            edge for edge in self.edges if self.edge_neighb_of_subset(edge, subset)
        }
        self.edges -= neighbouring_edges
        neighbouring_nodes = {
            edge.node2 if edge.node1 in subset else edge.node1
            for edge in neighbouring_edges
        }
        return neighbouring_nodes

    def get_rel_neighbourhood(self, from_clauses: ClauseSet, distance: int):

        neighbourhood = self.clauses_to_nodes(from_clauses)
        for _ in range(2 * distance - 1):
            new_neighbours = self.get_neighbours(neighbourhood)
            neighbourhood |= new_neighbours

        clauses = self.nodes_to_clauses(neighbourhood)
        return clauses
