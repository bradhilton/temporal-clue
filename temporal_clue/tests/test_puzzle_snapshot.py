import random
import json
from temporal_clue.scenario import TemporalClueScenario


def test_puzzle_creation_snapshot(snapshot):
    """Test that puzzle creation with a fixed seed matches the saved snapshot.

    This test ensures that puzzle creation is deterministic and consistent over time
    when using the same random seed and arguments.
    """
    # Set the random seed for deterministic results
    random.seed(42)

    # Create a scenario with fixed parameters
    scenario = TemporalClueScenario(
        min_players=1,
        max_players=2,
    )

    # Create a puzzle with the scenario
    puzzle = scenario.create_puzzle()

    # Get the puzzle data as a JSON-serializable dictionary
    puzzle_data = puzzle.json_data()

    # Assert that the puzzle matches the saved snapshot
    snapshot.assert_match(
        json.dumps(puzzle_data, indent=2, sort_keys=True), "puzzle_snapshot.json"
    )
