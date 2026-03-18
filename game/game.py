from collections import Counter
import random
from typing import Literal, get_args

Suspect = Literal[
    "Miss Scarlet",
    "Colonel Mustard",
    "Mrs. White",
    "Mr. Green",
    "Mrs. Peacock",
    "Professor Plum",
]

Weapon = Literal[
    "Candlestick",
    "Knife",
    "Lead Pipe",
    "Revolver",
    "Rope",
    "Wrench",
]

Room = Literal[
    "Study",
    "Hall",
    "Lounge",
    "Library",
    "Billiard Room",
    "Dining Room",
    "Conservatory",
    "Ballroom",
    "Kitchen",
]

Time = Literal[
    "Seven o'Clock",
    "Eight o'Clock",
    "Nine o'Clock",
    "Ten o'Clock",
    "Eleven o'Clock",
    "Twelve o'Clock",
]

Character = Literal["Mr. Boddy", *Suspect]

Piece = Literal[*Character, *Weapon]

Scenario = dict[Time, dict[Piece, Room]]

Solution = tuple[Suspect, Weapon, Room, Time]

Clue = tuple[Piece, Time, Room]

adjacent_rooms: dict[Room, tuple[Room, ...]] = {
    "Study": ("Hall", "Library", "Kitchen"),
    "Hall": ("Study", "Lounge"),
    "Lounge": ("Hall", "Dining Room", "Conservatory"),
    "Library": ("Study", "Billiard Room"),
    "Billiard Room": ("Library", "Conservatory"),
    "Dining Room": ("Lounge", "Kitchen"),
    "Conservatory": ("Lounge", "Billiard Room", "Ballroom"),
    "Ballroom": ("Conservatory", "Kitchen"),
    "Kitchen": ("Study", "Dining Room", "Ballroom"),
}
rooms: list[Room] = list(get_args(Room))
suspects: list[Suspect] = list(get_args(Suspect))
characters: list[Character] = ["Mr. Boddy", *get_args(Suspect)]
pieces: list[Piece] = [*characters, *get_args(Weapon)]
times: list[Time] = list(get_args(Time))
weapons: list[Weapon] = list(get_args(Weapon))

# Pilot counts from 1,000,000 raw valid scenarios.
_valid_time_room_counts: Counter[tuple[Time, Room]] = Counter({
    ("Seven o'Clock", "Study"): 19244,
    ("Seven o'Clock", "Hall"): 18245,
    ("Seven o'Clock", "Lounge"): 19411,
    ("Seven o'Clock", "Library"): 18376,
    ("Seven o'Clock", "Billiard Room"): 18357,
    ("Seven o'Clock", "Dining Room"): 18399,
    ("Seven o'Clock", "Conservatory"): 19453,
    ("Seven o'Clock", "Ballroom"): 18424,
    ("Seven o'Clock", "Kitchen"): 19014,
    ("Eight o'Clock", "Study"): 24154,
    ("Eight o'Clock", "Hall"): 11114,
    ("Eight o'Clock", "Lounge"): 23827,
    ("Eight o'Clock", "Library"): 13909,
    ("Eight o'Clock", "Billiard Room"): 13737,
    ("Eight o'Clock", "Dining Room"): 10929,
    ("Eight o'Clock", "Conservatory"): 23932,
    ("Eight o'Clock", "Ballroom"): 10988,
    ("Eight o'Clock", "Kitchen"): 23656,
    ("Nine o'Clock", "Study"): 24122,
    ("Nine o'Clock", "Hall"): 12611,
    ("Nine o'Clock", "Lounge"): 22798,
    ("Nine o'Clock", "Library"): 13836,
    ("Nine o'Clock", "Billiard Room"): 13656,
    ("Nine o'Clock", "Dining Room"): 12793,
    ("Nine o'Clock", "Conservatory"): 24366,
    ("Nine o'Clock", "Ballroom"): 12559,
    ("Nine o'Clock", "Kitchen"): 23048,
    ("Ten o'Clock", "Study"): 24358,
    ("Ten o'Clock", "Hall"): 13121,
    ("Ten o'Clock", "Lounge"): 23295,
    ("Ten o'Clock", "Library"): 13732,
    ("Ten o'Clock", "Billiard Room"): 13457,
    ("Ten o'Clock", "Dining Room"): 12852,
    ("Ten o'Clock", "Conservatory"): 24153,
    ("Ten o'Clock", "Ballroom"): 12925,
    ("Ten o'Clock", "Kitchen"): 23609,
    ("Eleven o'Clock", "Study"): 24958,
    ("Eleven o'Clock", "Hall"): 13497,
    ("Eleven o'Clock", "Lounge"): 24604,
    ("Eleven o'Clock", "Library"): 14043,
    ("Eleven o'Clock", "Billiard Room"): 13895,
    ("Eleven o'Clock", "Dining Room"): 13302,
    ("Eleven o'Clock", "Conservatory"): 24946,
    ("Eleven o'Clock", "Ballroom"): 13660,
    ("Eleven o'Clock", "Kitchen"): 24774,
    ("Twelve o'Clock", "Study"): 27662,
    ("Twelve o'Clock", "Hall"): 15272,
    ("Twelve o'Clock", "Lounge"): 26910,
    ("Twelve o'Clock", "Library"): 15662,
    ("Twelve o'Clock", "Billiard Room"): 15676,
    ("Twelve o'Clock", "Dining Room"): 15186,
    ("Twelve o'Clock", "Conservatory"): 27312,
    ("Twelve o'Clock", "Ballroom"): 15317,
    ("Twelve o'Clock", "Kitchen"): 26864,
})
_valid_time_room_weights: dict[tuple[Time, Room], float] = {
    key: min(_valid_time_room_counts.values()) / count
    for key, count in _valid_time_room_counts.items()
}

def random_scenario() -> Scenario:
    scenario: Scenario = {
        "Seven o'Clock": {
            **{piece: random.choice(rooms) for piece in pieces},
        },
    }
    for time, next_time in zip(times[:-1], times[1:]):
        scenario[next_time] = scenario[time].copy()
        shuffled_characters = characters.copy()
        random.shuffle(shuffled_characters)
        for character in shuffled_characters:
            if random.random() < 0.33:
                # character stays in the same room
                continue
            scenario[next_time][character] = random.choice(adjacent_rooms[scenario[time][character]])
            if character == "Mr. Boddy" or random.random() < 0.33:
                continue
            # character moves weapon if possible
            shuffled_weapons = weapons.copy()
            random.shuffle(shuffled_weapons)
            for weapon in shuffled_weapons:
                if scenario[time][weapon] != scenario[time][character]:
                    continue
                scenario[next_time][weapon] = scenario[next_time][character]
                break
    return scenario


def get_solution(scenario: Scenario) -> Solution | None:
    solution: Solution | None = None
    for time in times:
        for room in rooms:
            if scenario[time]["Mr. Boddy"] != room:
                continue
            suspects_here: list[Suspect] = [suspect for suspect in suspects if scenario[time][suspect] == room]
            weapons_here: list[Weapon] = [weapon for weapon in weapons if scenario[time][weapon] == room]
            if len(suspects_here) != 1 or len(weapons_here) != 1:
                continue
            if solution is not None:
                return None
            solution = (suspects_here[0], weapons_here[0], room, time)
    return solution


def random_valid_scenario_and_solution() -> tuple[Scenario, Solution]:
    while True:
        scenario = random_scenario()
        solution = get_solution(scenario)
        if solution is None:
            continue
        _, _, room, time = solution
        if random.random() < _valid_time_room_weights[time, room]:
            return scenario, solution
