"""Unit test for agent/state_encoder.py — thalamic attention gating."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.state_encoder import StateEncoder


class FakeNode:
    def __init__(self, category, value, confidence, step):
        self.category = category
        self.value = value
        self.confidence = confidence
        self.step = step


class FakeGraph:
    def __init__(self):
        self.nodes = {}

    def add(self, key, value, category, confidence, step):
        self.nodes[key] = FakeNode(category, value, confidence, step)


class FakeWorkbench:
    def __init__(self):
        self.graph = FakeGraph()


def test_state_encoder_without_hints():
    wb = FakeWorkbench()
    wb.graph.add("cpu_cores", "22", "system", 0.9, 100)
    wb.graph.add("os", "linux", "system", 0.8, 90)
    wb.graph.add("old_fact", "x", "general", 0.5, 10)

    enc = StateEncoder(workbench=wb)
    enc.set_step(110)
    enc.set_mode("EXPLORE")
    text = enc.get_state_text()
    assert "cpu_cores" in text
    assert "os" in text
    print("PASSED: StateEncoder without hints works")


def test_state_encoder_with_hints():
    wb = FakeWorkbench()
    wb.graph.add("cpu_cores", "22", "system", 0.5, 100)
    wb.graph.add("os", "linux", "system", 0.5, 100)
    wb.graph.add("network_iface", "eth0", "network", 0.5, 100)

    enc = StateEncoder(workbench=wb)
    enc.set_step(110)
    enc.set_mode("EXPLORE")

    # Without hints, order is determined by category match and confidence.
    # With hints, "network_iface" should be boosted to appear earlier.
    hints = {"network_iface": 0.9}

    text_no_hints = enc.get_state_text()
    text_with_hints = enc.get_state_text(salience_hints=hints)

    # Both should contain all facts
    assert "network_iface" in text_with_hints
    # The boosted fact should appear earlier in the with-hints text
    pos_no_hints = text_no_hints.find("network_iface")
    pos_with_hints = text_with_hints.find("network_iface")
    assert pos_with_hints <= pos_no_hints, (
        f"boosted fact should appear earlier or same: "
        f"no_hints={pos_no_hints}, with_hints={pos_with_hints}"
    )
    print("PASSED: salience_hints boosts high-salience fact earlier")


def test_apply_attention():
    enc = StateEncoder()
    scores = [("a", 0.3), ("b", 0.9), ("c", 0.5)]
    top = enc.apply_attention(scores, top_k=2)
    assert len(top) == 2
    assert top[0][0] == "b"
    assert top[1][0] == "c"
    print("PASSED: apply_attention sorts and truncates")


if __name__ == "__main__":
    test_state_encoder_without_hints()
    test_state_encoder_with_hints()
    test_apply_attention()
    print("All StateEncoder tests PASSED")
