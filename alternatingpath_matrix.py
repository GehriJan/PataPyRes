from clausesets import ClauseSet
from alternatingpath_abstract import RelevanceGraph
from fofspec import FOFSpec
from literals import Literal
from clauses import Clause
from unification import mgu
from literals import literalList2String
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.sparse.csgraph import dijkstra


class MatrixRelevanceGraph(RelevanceGraph):

    def __init__(self, clause_set):
        self.construct_graph(clause_set)
        return

    def construct_graph(self, clause_set):
        self.nodes = self.construct_nodes(clause_set)
        self.adjacency_matrix = self.construct_matrix()

    def clauses_to_nodes(self, clauses):
        return np.nonzero(np.isin(self.nodes[:, 0], clauses.clauses))[0]

    def nodes_to_clauses(self, nodes):
        return ClauseSet(set(self.nodes[nodes, 0]))

    def construct_nodes(self, clauses):
        nodes = []
        for clause in clauses.clauses:
            for literal in clause.literals:
                for _ in range(2):
                    nodes.append((clause, literal, literal.negative, literal.atom))
        return np.array(nodes, dtype=object)

    def construct_matrix(self):
        return np.fromfunction(
            self.is_connected, shape=2 * [self.nodes.shape[0]], dtype=int
        )

    def is_connected(self, i, j):
        nodes_i = self.nodes[i, :]
        nodes_j = self.nodes[j, :]

        upper_left_triangle = j > i
        different_direction = (np.indices(i.shape).sum(axis=0) % 2).astype(bool)

        same_clause = nodes_i[:, :, 0] == nodes_j[:, :, 0]
        different_literal = nodes_i[:, :, 1] != nodes_j[:, :, 1]

        different_literal_sign = nodes_i[:, :, 2] != nodes_j[:, :, 2]
        mgu_exists_vectorized = np.vectorize(lambda x, y: mgu(x, y) is not None)
        mgu_exists = mgu_exists_vectorized(nodes_i[:, :, 3], nodes_j[:, :, 3])

        in_clause_connected = same_clause & different_literal
        between_clause_connected = ~same_clause & different_literal_sign & mgu_exists
        return (
            upper_left_triangle
            & different_direction
            & (in_clause_connected | between_clause_connected)
        )

    def get_rel_neighbourhood(self, from_clauses, distance):
        from_nodes = self.clauses_to_nodes(from_clauses)
        distances_per_node = dijkstra(
            csgraph=self.adjacency_matrix,
            indices=from_nodes,
            directed=False,
            unweighted=True,
            limit=max(2 * distance - 1, 0),
        )
        distance_to_set = np.min(distances_per_node, axis=0)
        nodes_neighbourhood = np.argwhere(np.isfinite(distance_to_set)).flatten()
        clauses_neighbourhood = self.nodes_to_clauses(nodes_neighbourhood)
        return clauses_neighbourhood

    ####################################
    # PLOTTING FUNCTIONS
    ####################################

    def to_mermaid(self, path: str) -> str:
        G = self.create_nx_graph()
        edge_x, edge_y = self.get_edge_coordinates(G)
        edge_trace = self.create_edge_trace(edge_x, edge_y)
        node_x, node_y = self.get_node_coordinates(G)
        node_trace = self.create_node_trace(node_x, node_y)
        self.node_to_rel_distance()
        fig = self.create_figure(edge_trace, node_trace)
        fig.show(renderer="browser")

    def create_nx_graph(self):
        G = nx.Graph()
        # Get edges
        rows, cols = np.where(self.adjacency_matrix == True)
        edges = zip(rows.tolist(), cols.tolist())
        # Add edges
        G.add_edges_from(edges)
        # Add position attribute
        pos = nx.spring_layout(G)
        nx.set_node_attributes(G, pos, "pos")

        return G

    def get_node_labels(self):
        return [
            f"clause: {literalList2String(clause.literals)}\n"
            + f"literal: {literal}\n"
            + f"direction: {"out" if index%2==0 else "in"}"
            for index, (clause, literal) in enumerate(self.nodes[:, 0:2])
        ]

    def get_edge_coordinates(self, G):
        edge_x = []
        edge_y = []
        for start, end in G.edges():
            x0, y0 = G.nodes[start]["pos"]
            x1, y1 = G.nodes[end]["pos"]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        return edge_x, edge_y

    def create_edge_trace(self, edge_x, edge_y):
        return go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

    def get_node_coordinates(self, G):
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = G.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)
        return node_x, node_y

    def create_node_trace(self, node_x, node_y):
        return go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=self.get_node_labels(),
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                reversescale=True,
                color=self.node_to_rel_distance(),
                size=10,
                colorbar=dict(
                    thickness=15,
                    title=dict(text="Node distance", side="right"),
                    xanchor="left",
                ),
                line_width=2,
            ),
        )

    def create_figure(self, edge_trace, node_trace):
        return go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text="<br>Network graph made with Python", font=dict(size=16)
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

    def node_to_rel_distance(self):
        from_clauses = ClauseSet(
            {
                clause
                for clause in self.nodes[:, 0]
                if clause.type == "negated_conjecture"
            }
        )
        from_nodes = self.clauses_to_nodes(from_clauses)
        distances_per_node = dijkstra(
            csgraph=self.adjacency_matrix,
            indices=from_nodes,
            directed=False,
            unweighted=True,
        )
        distance_to_set = np.min(distances_per_node, axis=0)
        return distance_to_set
