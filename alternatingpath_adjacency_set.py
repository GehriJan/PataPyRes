from clauses import Clause
from clausesets import ClauseSet
from literals import Literal
from unification import mgu
from collections import defaultdict
from alternatingpath_abstract import RelevanceGraph
from typing import Union
from copy import deepcopy


class Node:
    def __init__(self, literal: Literal, clause: Clause) -> None:
        self.literal: Literal = literal
        self.clause: Clause = clause
        self.neighbours: set[Node] = set()

    def __repr__(self) -> str:
        return f"<{self.clause.name},{self.literal}>"


class AdjacencySetRelevanceGraph(RelevanceGraph):

    def construct_graph(self, clause_set: ClauseSet) -> None:
        self.out_nodes = self.construct_nodes(clause_set)
        self.in_nodes = self.construct_nodes(clause_set)
        self.construct_inclause_edges()
        self.construct_betweenclause_edges()

    @staticmethod
    def construct_nodes(clause_set: ClauseSet):
        return {
            Node(literal, clause)
            for clause in clause_set.clauses
            for literal in clause.literals
        }

    def construct_inclause_edges(self):
        for in_node in self.in_nodes:
            in_node.neighbours = {
                out_node
                for out_node in self.out_nodes
                if in_node.clause == out_node.clause
                and in_node.literal != out_node.literal
            }

    def construct_betweenclause_edges(self):
        for out_node in self.out_nodes:
            out_node.neighbours = {
                in_node
                for in_node in self.in_nodes
                if (out_node.literal.negative != in_node.literal.negative)
                and (mgu(out_node.literal.atom, in_node.literal.atom) != None)
            }

    @staticmethod
    def nodes_to_clauses(nodes):
        return ClauseSet({node.clause for node in nodes})

    def clauses_to_nodes(self, clauses: ClauseSet):
        return {node for node in self.out_nodes if node.clause in clauses.clauses}

    def extend_neighbourhood(self, outer_nodes: set[Node], neighbourhood: set[Node]):
        return {
            neighbour for node in outer_nodes for neighbour in node.neighbours
        } - neighbourhood

    def get_rel_neighbourhood(self, from_clauses: ClauseSet, distance: Union[int, str]):
        neighbourhood = self.clauses_to_nodes(from_clauses)
        new_neighbours = neighbourhood.copy()
        search_range = (2 * distance - 1) if distance != "n" else 2 * len(self.in_nodes)
        for _ in range(search_range):
            new_neighbours = self.extend_neighbourhood(new_neighbours, neighbourhood)
            if not new_neighbours:
                break
            neighbourhood |= new_neighbours
        return self.nodes_to_clauses(neighbourhood)
