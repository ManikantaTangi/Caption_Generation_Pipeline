"""
Stage 5 - Ontology Loader
============================
Purpose
    Load the agricultural-domain ontology (configs/ontology.yaml -- class
    hierarchy, object properties, axioms) into an in-memory, queryable
    form. This is the formal semantic backbone that lets the pipeline
    talk about "Rice is-a CerealCrop is-a Crop is-a Vegetation" rather
    than only flat class labels, and lets Stage 8 check generated facts
    against `axioms`.

Algorithm
    NetworkX `DiGraph` for the `subClassOf` hierarchy (nodes = classes,
    edges = subClassOf, direction child->parent so `nx.ancestors` gives
    the full superclass chain in O(V+E)); object properties and axioms
    kept as structured lists for direct lookup. Loading is O(V+E), V,E
    small (fixed ontology size).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ObjectProperty:
    name: str
    domain: str
    range: str
    symmetric: bool = False


@dataclass
class Axiom:
    subject: str
    predicate: str
    object: str


class OntologyLoader:
    """Loads the YAML ontology into a queryable class hierarchy graph."""

    def __init__(self, ontology_path: str) -> None:
        with open(ontology_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        self.hierarchy = nx.DiGraph()
        for cls_name, info in raw["classes"].items():
            self.hierarchy.add_node(cls_name)
            parent = info.get("subClassOf")
            if parent:
                self.hierarchy.add_edge(cls_name, parent)  # child -> parent

        self.object_properties: Dict[str, ObjectProperty] = {
            p["name"]: ObjectProperty(p["name"], p["domain"], p["range"], p.get("symmetric", False))
            for p in raw.get("object_properties", [])
        }
        self.axioms: List[Axiom] = [
            Axiom(a["subject"], a["predicate"], a["object"]) for a in raw.get("axioms", [])
        ]
        logger.info("Loaded ontology: %d classes, %d object properties, %d axioms.",
                    self.hierarchy.number_of_nodes(), len(self.object_properties), len(self.axioms))

    def superclasses(self, class_name: str) -> List[str]:
        """Full ancestor chain (excluding self), root-most last."""
        if class_name not in self.hierarchy:
            return []
        return list(nx.dfs_preorder_nodes(self.hierarchy, class_name))[1:]

    def is_subclass_of(self, child: str, ancestor: str) -> bool:
        if child not in self.hierarchy or ancestor not in self.hierarchy:
            return False
        return ancestor in nx.descendants(self.hierarchy, child) or child == ancestor

    def has_axiom(self, subject: str, predicate: str, obj: str) -> bool:
        return any(a.subject == subject and a.predicate == predicate and a.object == obj for a in self.axioms)
