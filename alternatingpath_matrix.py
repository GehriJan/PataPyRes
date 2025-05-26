from clausesets import ClauseSet
from alternatingpath_abstract import RelevanceGraph
from fofspec import FOFSpec
from literals import Literal
from clauses import Clause
import pandas as pd

class MatrixRelevanceGraph(RelevanceGraph):

    def __init__(self, clause_set):


    def construct_graph(self, clause_set):
        return super().construct_graph(clause_set)


    def get_rel_neighbourhood(self, from_clauses, distance):
        return super().get_rel_neighbourhood(from_clauses, distance)

