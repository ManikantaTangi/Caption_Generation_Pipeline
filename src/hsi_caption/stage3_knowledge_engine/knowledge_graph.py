"""
Stage 3 - Knowledge Graph Construction
==========================================
Purpose
    Materialise the spectral library + retrieval results as an explicit
    graph structure (materials -> categories, materials <-> materials by
    spectral similarity) so later stages can reason over *relationships*
    ("Rice and Water are similarity-linked") instead of only flat
    material lists. This is the graph Stage 8's KnowledgeGraphVerification
    queries against.

Input
    `SpectralLibrary` (this stage) + a patch's `List[MaterialMatch]` from
    KnowledgeRetrieval.

Algorithm
    NetworkX `DiGraph`: one node per material, one node per category,
    `belongsTo` edges material->category, and `similarTo` edges between
    material pairs whose library-spectra SAM-similarity exceeds
    `kg_similarity_threshold` (captures materials that are easily
    confused, e.g. two soybean cultivars). Per-patch query nodes are
    added transiently and connected to their top-k retrieved materials
    with weighted `matchedTo` edges, then queried for their ids (not
    persisted into the shared graph, to keep it a static reference graph).
    Complexity: O(M^2) to build the static similarity edges (M small);
    O(k) per patch query.
"""
from __future__ import annotations

import logging
from typing import List

import networkx as nx
import numpy as np

from hsi_caption.datatypes import MaterialMatch
from hsi_caption.stage3_knowledge_engine.spectral_library import SpectralLibrary

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds and queries the static material/category knowledge graph."""

    def __init__(self, library: SpectralLibrary, similarity_threshold: float) -> None:
        self.library = library
        self.similarity_threshold = similarity_threshold
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        matrix, names = self.library.matrix()
        categories = {self.library.category_of(n) for n in names}
        for cat in categories:
            g.add_node(f"category::{cat}", type="category")
        for name in names:
            cat = self.library.category_of(name)
            g.add_node(f"material::{name}", type="material")
            g.add_edge(f"material::{name}", f"category::{cat}", relation="belongsTo")

        norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        sim_matrix = norm @ norm.T
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j and sim_matrix[i, j] >= self.similarity_threshold:
                    g.add_edge(f"material::{names[i]}", f"material::{names[j]}",
                               relation="similarTo", weight=float(sim_matrix[i, j]))
        logger.info("Built knowledge graph: %d nodes, %d edges.", g.number_of_nodes(), g.number_of_edges())
        return g

    def query_matched_node_ids(self, matches: List[MaterialMatch]) -> List[str]:
        return [f"material::{m.material_name}" for m in matches if f"material::{m.material_name}" in self.graph]

    def related_materials(self, material_name: str) -> List[str]:
        node = f"material::{material_name}"
        if node not in self.graph:
            return []
        return [
            n.split("::", 1)[1] for n in self.graph.successors(node)
            if self.graph.edges[node, n]["relation"] == "similarTo"
        ]
