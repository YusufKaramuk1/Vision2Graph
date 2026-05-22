"""Testler icin ortak sentetik yol grafigi.

Gercek DeepGlobe verisi ve model checkpoint'i depoda olmadigi icin testler
elde uretilen kucuk bir grafik uzerinde calisir; hizli ve veriden bagimsizdir.
"""
import math

import networkx as nx
import pytest

# (x, y) koordinatlari: 0-1-2-3-4 yildizi (merkez 1) + ayri bir 5-6 bileseni
_COORDS = {
    0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (10, 10),
    4: (10, -10), 5: (100, 100), 6: (130, 100),
}
_EDGES = [(0, 1), (1, 2), (1, 3), (1, 4), (5, 6)]


@pytest.fixture
def sample_graph():
    """Kucuk sentetik yol grafigi.

    Node 1 merkezdedir (derece 4) ve bir kesim noktasidir: kaldirilinca 0/2/3/4
    ayri ayri kopar. 5-6 bagimsiz bir bilesendir. Grafik bir orman oldugu icin
    tum kenarlar koprudur. Toplam 7 kavsak, 5 yol, 2 bilesen.
    """
    graph = nx.Graph()
    for node, (x, y) in _COORDS.items():
        graph.add_node(node, x=float(x), y=float(y))
    for u, v in _EDGES:
        length = math.hypot(_COORDS[u][0] - _COORDS[v][0],
                            _COORDS[u][1] - _COORDS[v][1])
        graph.add_edge(u, v, length=length, weight=length, pts=None)
    return graph
