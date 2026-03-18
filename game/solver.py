from collections import deque
from itertools import pairwise

from ortools.sat.python import cp_model

from .game import (
    Clue,
    Solution,
    adjacent_rooms,
    characters,
    pieces,
    rooms,
    suspects,
    times,
    weapons,
)

_SUSPECT_SET = frozenset(suspects)
_WEAPON_SET = frozenset(weapons)
_TIME_INDEX = {time: i for i, time in enumerate(times)}
_ROOM_DISTANCES: dict[str, dict[str, int]] = {}
for start_room in rooms:
    _ROOM_DISTANCES[start_room] = {start_room: 0}
    frontier = deque([start_room])
    while frontier:
        room = frontier.popleft()
        for next_room in adjacent_rooms[room]:
            if next_room in _ROOM_DISTANCES[start_room]:
                continue
            _ROOM_DISTANCES[start_room][next_room] = _ROOM_DISTANCES[start_room][room] + 1
            frontier.append(next_room)


def _reachable_rooms(start_rooms: set[str]) -> set[str]:
    return {
        next_room
        for room in start_rooms
        for next_room in (room, *adjacent_rooms[room])
    }


def _all(
    model: cp_model.CpModel,
    *bool_vars: cp_model.IntVar,
    name: str,
) -> cp_model.IntVar:
    var = model.NewBoolVar(name)
    model.Add(sum(bool_vars) == len(bool_vars)).OnlyEnforceIf(var)
    model.Add(sum(bool_vars) < len(bool_vars)).OnlyEnforceIf(var.Not())
    return var


def _sum_is(
    model: cp_model.CpModel,
    bool_vars: list[cp_model.IntVar],
    value: int,
    *,
    name: str,
) -> cp_model.IntVar:
    var = model.NewBoolVar(name)
    model.Add(sum(bool_vars) == value).OnlyEnforceIf(var)
    model.Add(sum(bool_vars) != value).OnlyEnforceIf(var.Not())
    return var


def _clue_rules_out_solution(clue: Clue, solution: Solution) -> bool:
    piece, clue_time, clue_room = clue
    suspect, weapon, room, time = solution

    clue_time_index = _TIME_INDEX[clue_time]
    time_index = _TIME_INDEX[time]

    if piece == "Mr. Boddy":
        if clue_time_index > time_index:
            return False
        if clue_time == time:
            return clue_room != room
        return _ROOM_DISTANCES[clue_room][room] > time_index - clue_time_index

    if clue_time == time:
        if piece in _SUSPECT_SET:
            return clue_room != room if piece == suspect else clue_room == room
        if piece in _WEAPON_SET:
            return clue_room != room if piece == weapon else clue_room == room

    time_delta = abs(clue_time_index - time_index)
    if piece in {suspect, weapon}:
        return _ROOM_DISTANCES[clue_room][room] > time_delta
    return False


def _piece_time_room_domains(
    clues: list[Clue],
) -> dict[str, dict[str, set[str]]]:
    transitions = list(pairwise(times))
    domains = {
        piece: {time: set(rooms) for time in times}
        for piece in pieces
        if piece != "Mr. Boddy"
    }
    for piece, time, room in clues:
        if piece == "Mr. Boddy":
            continue
        domains[piece][time] &= {room}

    changed = True
    while changed:
        changed = False
        for piece in domains:
            for prev_time, next_time in transitions:
                next_rooms = domains[piece][next_time] & _reachable_rooms(domains[piece][prev_time])
                if next_rooms != domains[piece][next_time]:
                    domains[piece][next_time] = next_rooms
                    changed = True

                prev_rooms = domains[piece][prev_time] & _reachable_rooms(domains[piece][next_time])
                if prev_rooms != domains[piece][prev_time]:
                    domains[piece][prev_time] = prev_rooms
                    changed = True
    return domains


def valid_solutions(clues: list[Clue]) -> list[Solution]:
    model = cp_model.CpModel()
    transitions = list(pairwise(times))
    routes = [
        (prev_room, next_room)
        for prev_room in rooms
        for next_room in adjacent_rooms[prev_room]
    ]
    piece_time_room_domains = _piece_time_room_domains(clues)

    piece_time_room_vars = {
        piece: {
            time: {
                room: model.NewBoolVar(f"{piece} @ {room} / {time}")
                for room in rooms
            }
            for time in times
        }
        for piece in pieces
    }

    for piece in pieces:
        for time in times:
            model.AddExactlyOne(piece_time_room_vars[piece][time].values())
            if piece == "Mr. Boddy":
                continue
            for room in rooms:
                if room not in piece_time_room_domains[piece][time]:
                    model.Add(piece_time_room_vars[piece][time][room] == 0)

    for mover in [*characters, *weapons]:
        for prev_time, next_time in transitions:
            for prev_room in rooms:
                for next_room in rooms:
                    if next_room == prev_room or next_room in adjacent_rooms[prev_room]:
                        continue
                    model.AddAtMostOne(
                        piece_time_room_vars[mover][prev_time][prev_room],
                        piece_time_room_vars[mover][next_time][next_room],
                    )

    suspect_move_vars = {
        suspect: {
            transition: {
                route: _all(
                    model,
                    piece_time_room_vars[suspect][transition[0]][route[0]],
                    piece_time_room_vars[suspect][transition[1]][route[1]],
                    name=(
                        f"{suspect} moves from {route[0]} to {route[1]} "
                        f"between {transition[0]} and {transition[1]}"
                    ),
                )
                for route in routes
            }
            for transition in transitions
        }
        for suspect in suspects
    }
    weapon_move_vars = {
        weapon: {
            transition: {
                route: _all(
                    model,
                    piece_time_room_vars[weapon][transition[0]][route[0]],
                    piece_time_room_vars[weapon][transition[1]][route[1]],
                    name=(
                        f"{weapon} moves from {route[0]} to {route[1]} "
                        f"between {transition[0]} and {transition[1]}"
                    ),
                )
                for route in routes
            }
            for transition in transitions
        }
        for weapon in weapons
    }

    for transition in transitions:
        for route in routes:
            model.Add(
                sum(weapon_move_vars[weapon][transition][route] for weapon in weapons)
                <= sum(suspect_move_vars[suspect][transition][route] for suspect in suspects)
            )

    one_suspect_vars = {
        time: {
            room: _sum_is(
                model,
                [piece_time_room_vars[suspect][time][room] for suspect in suspects],
                1,
                name=f"exactly one suspect @ {room} / {time}",
            )
            for room in rooms
        }
        for time in times
    }
    one_weapon_vars = {
        time: {
            room: _sum_is(
                model,
                [piece_time_room_vars[weapon][time][room] for weapon in weapons],
                1,
                name=f"exactly one weapon @ {room} / {time}",
            )
            for room in rooms
        }
        for time in times
    }

    murder_cell_vars = {
        (time, room): _all(
            model,
            piece_time_room_vars["Mr. Boddy"][time][room],
            one_suspect_vars[time][room],
            one_weapon_vars[time][room],
            name=f"murder setup @ {room} / {time}",
        )
        for time in times
        for room in rooms
    }
    solution_vars = {
        (suspect, weapon, room, time): _all(
            model,
            murder_cell_vars[(time, room)],
            piece_time_room_vars[suspect][time][room],
            piece_time_room_vars[weapon][time][room],
            name=f"solution: {suspect} / {weapon} / {room} / {time}",
        )
        for time in times
        for room in rooms
        for suspect in suspects
        for weapon in weapons
    }
    for clue in clues:
        piece, time, room = clue
        if piece == "Mr. Boddy":
            model.AddBoolOr(
                [
                    piece_time_room_vars["Mr. Boddy"][time][room],
                    *[
                        solution_var
                        for solution, solution_var in solution_vars.items()
                        if _TIME_INDEX[solution[3]] < _TIME_INDEX[time]
                    ],
                ]
            )
            continue
        model.Add(piece_time_room_vars[piece][time][room] == 1)
        for solution, solution_var in solution_vars.items():
            if _clue_rules_out_solution(clue, solution):
                model.Add(solution_var == 0)
    for solution, solution_var in solution_vars.items():
        suspect, weapon, room, time = solution
        if room not in piece_time_room_domains[suspect][time]:
            model.Add(solution_var == 0)
            continue
        if room not in piece_time_room_domains[weapon][time]:
            model.Add(solution_var == 0)
            continue
        if any(
            piece_time_room_domains[other_suspect][time] == {room}
            for other_suspect in suspects
            if other_suspect != suspect
        ):
            model.Add(solution_var == 0)
            continue
        if any(
            piece_time_room_domains[other_weapon][time] == {room}
            for other_weapon in weapons
            if other_weapon != weapon
        ):
            model.Add(solution_var == 0)
    model.AddExactlyOne(solution_vars.values())

    solver = cp_model.CpSolver()
    solutions: list[Solution] = []
    while solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = next(
            solution
            for solution, solution_var in solution_vars.items()
            if solver.BooleanValue(solution_var)
        )
        solutions.append(solution)
        model.Add(solution_vars[solution] == 0)

    room_order = {room: i for i, room in enumerate(rooms)}
    time_order = {time: i for i, time in enumerate(times)}
    suspect_order = {suspect: i for i, suspect in enumerate(suspects)}
    weapon_order = {weapon: i for i, weapon in enumerate(weapons)}
    return sorted(
        solutions,
        key=lambda solution: (
            time_order[solution[3]],
            room_order[solution[2]],
            suspect_order[solution[0]],
            weapon_order[solution[1]],
        ),
    )
