"""Unit test for agent/salience.py — SalienceSignal."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.salience import SalienceSignal


def test_salience_range():
    sal = SalienceSignal()
    for _ in range(10):
        s = sal.update(rnd_novelty=0.1, wm_surprise=0.2, success=True, reward=0.5)
        assert 0.0 <= s <= 1.0, f"salience out of range: {s}"
    print("PASSED: salience in [0, 1]")


def test_window_stats():
    sal = SalienceSignal(window_size=5)
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for v in values:
        # 构造一个能产出大致 v 的输入组合
        sal.update(rnd_novelty=v * 0.5, wm_surprise=0.0, success=False, reward=0.0)
    stats = sal.get_stats()
    assert stats["window_len"] == 5, f"window size mismatch: {stats['window_len']}"
    assert "mean" in stats and "max" in stats and "min" in stats
    print(f"PASSED: window stats mean={stats['mean']:.2f} max={stats['max']:.2f} min={stats['min']:.2f}")


def test_empty_window():
    sal = SalienceSignal()
    assert sal.recent_mean() == 0.0
    assert sal.recent_max() == 0.0
    assert sal.recent_min() == 0.0
    print("PASSED: empty window returns 0")


if __name__ == "__main__":
    test_salience_range()
    test_window_stats()
    test_empty_window()
    print("All SalienceSignal tests PASSED")
