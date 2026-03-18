import random

from game.game import (
    Clue,
    Scenario,
    Solution,
    pieces,
    rooms,
    random_valid_scenario_and_solution,
    times,
)
from game.solver import valid_solutions

MIN_RANDOM_SUBSET_SIZE = len(pieces) * len(times) // 6


def _all_clues(scenario: Scenario) -> list[Clue]:
    return [
        (piece, time, scenario[time][piece])
        for time in times
        for piece in pieces
    ]


def _assert_solution_preserved(clues: list[Clue], solution: Solution) -> list[Solution]:
    solutions = valid_solutions(clues)
    assert solution in solutions
    return solutions


def test_valid_solutions_with_random_subsets_preserve_valid_solution():
    rng = random.Random(0)

    for subset_size in (
        MIN_RANDOM_SUBSET_SIZE * 2,
        MIN_RANDOM_SUBSET_SIZE * 3,
        MIN_RANDOM_SUBSET_SIZE * 4,
    ):
        scenario, solution = random_valid_scenario_and_solution()
        clues = rng.sample(_all_clues(scenario), subset_size)

        _assert_solution_preserved(clues, solution)


def test_valid_solutions_with_all_clues_returns_exact_solution():
    for _ in range(3):
        scenario, solution = random_valid_scenario_and_solution()

        assert valid_solutions(_all_clues(scenario)) == [solution]


def test_valid_solutions_rejects_non_adjacent_weapon_move():
    assert valid_solutions(
        [
            ("Knife", "Seven o'Clock", "Study"),
            ("Knife", "Eight o'Clock", "Ballroom"),
        ]
    ) == []


def test_valid_solutions_treats_late_boddy_clues_as_vacuous():
    scenario = None
    solution = None
    for _ in range(20):
        candidate_scenario, candidate_solution = random_valid_scenario_and_solution()
        if candidate_solution[3] != "Twelve o'Clock":
            scenario = candidate_scenario
            solution = candidate_solution
            break

    assert scenario is not None
    assert solution is not None

    wrong_room = next(
        room for room in rooms if room != scenario["Twelve o'Clock"]["Mr. Boddy"]
    )
    clues = [
        clue
        for clue in _all_clues(scenario)
        if clue[0] != "Mr. Boddy"
    ]
    solutions_without_wrong_clue = valid_solutions(clues)
    clues.append(("Mr. Boddy", "Twelve o'Clock", wrong_room))
    solutions_with_wrong_clue = valid_solutions(clues)

    assert solution in solutions_without_wrong_clue
    assert solution in solutions_with_wrong_clue
    assert [
        candidate_solution
        for candidate_solution in solutions_with_wrong_clue
        if candidate_solution[3] != "Twelve o'Clock"
    ] == [
        candidate_solution
        for candidate_solution in solutions_without_wrong_clue
        if candidate_solution[3] != "Twelve o'Clock"
    ]
