"""Unit test for agent/habit.py — HabitSystem."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.habit import HabitSystem


def test_habit_formation_and_suggestion():
    habit = HabitSystem(confidence_threshold=0.7, min_success_count=3)
    intent = 0  # e.g., OBSERVE
    params = {"path": "/etc/hostname"}
    # 5 successes + 1 failure -> 5/6 = 0.833 > 0.7, success_count=5 >= 3
    for step in range(5):
        habit.register(intent, params, success=True, reward=0.5, step=step)
    habit.register(intent, params, success=False, reward=-0.2, step=5)
    suggested = habit.suggest(intent)
    assert suggested is not None, "expected habit suggestion after enough successes"
    assert suggested["path"] == "/etc/hostname"
    stats = habit.get_stats()
    assert stats["n_habits"] == 1
    assert stats["top"][0]["confidence"] > 0.7
    print("PASSED: habit forms and suggests after 5 successes + 1 failure")


def test_habit_no_suggestion_below_threshold():
    habit = HabitSystem(confidence_threshold=0.7, min_success_count=3)
    intent = 0
    params = {"path": "/tmp"}
    # 3 failures -> confidence 0, should not suggest
    for step in range(3):
        habit.register(intent, params, success=False, reward=-0.5, step=step)
    suggested = habit.suggest(intent)
    assert suggested is None, "should not suggest when confidence is 0"
    print("PASSED: no suggestion below threshold")


def test_habit_min_success_threshold():
    habit = HabitSystem(confidence_threshold=0.7, min_success_count=3)
    intent = 1
    params = {"custom_args": ["uname", "-a"]}
    # 2 successes, 0 failures -> confidence 1.0 but success_count=2 < 3
    for step in range(2):
        habit.register(intent, params, success=True, reward=0.5, step=step)
    suggested = habit.suggest(intent)
    assert suggested is None, "should not suggest before min_success_count"
    print("PASSED: min_success_count gate works")


if __name__ == "__main__":
    test_habit_formation_and_suggestion()
    test_habit_no_suggestion_below_threshold()
    test_habit_min_success_threshold()
    print("All HabitSystem tests PASSED")
