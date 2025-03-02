from ortools.sat.python import cp_model
from typing import Hashable, TYPE_CHECKING, TypeVar

from .types import Motive, Room, Suspect, Time, Weapon

if TYPE_CHECKING:
    from .scenario import TemporalClueScenario

H = TypeVar("H", bound=Hashable)
T = TypeVar("T")


class TemporalClueCpModel(cp_model.CpModel):
    """CP-SAT model with helper methods for Temporal Clue puzzle generation."""

    def __init__(self, scenario: "TemporalClueScenario") -> None:
        super().__init__()
        self._all_vars: dict[frozenset[cp_model.IntVar], cp_model.IntVar] = {}
        self._any_vars: dict[frozenset[cp_model.IntVar], cp_model.IntVar] = {}
        self._sum_vars: dict[
            tuple[frozenset[cp_model.IntVar], int], cp_model.IntVar
        ] = {}
        self._variable_element_time_room_vars: dict[
            tuple[
                dict[Suspect | Weapon, dict[Time, dict[Room, cp_model.IntVar]]],
                dict[Suspect | Weapon, cp_model.IntVar],
                Time,
                Room,
            ],
            cp_model.IntVar,
        ] = {}
        self.scenario = scenario

        # Create variables for suspect locations at each time
        self.suspect_time_room_vars = {
            suspect: {
                time: self.categorical_vars(scenario.rooms, prefix=f"{suspect} {time} ")
                for time in scenario.times
            }
            for suspect in scenario.suspects
        }

        # Create variables for suspect motives
        self.suspect_motive_vars = {
            suspect: self.categorical_vars(
                scenario.motives, prefix=f"{suspect} motive "
            )
            for suspect in scenario.suspects
        }
        self.motive_suspect_vars = {
            motive: {
                suspect: self.suspect_motive_vars[suspect][motive]
                for suspect in scenario.suspects
            }
            for motive in scenario.motives
        }
        for motive, suspects in scenario.motive_suspects.items():
            self.add(
                sum(
                    self.suspect_motive_vars[suspect][motive]
                    for suspect in scenario.suspects
                )
                == len(suspects)
            )

        # Create variables for weapon locations at each time
        self.weapon_time_room_vars = {
            weapon: {
                time: self.categorical_vars(scenario.rooms, prefix=f"{weapon} {time} ")
                for time in scenario.times
            }
            for weapon in scenario.weapons
        }

        # Create variables for Mr. Boddy's time rooms
        self.mr_boddy_time_room_vars = {
            time: self.categorical_vars(scenario.rooms, prefix=f"Mr. Boddy {time} ")
            for time in scenario.times
        }

        # Variables for character locations at each time
        self.character_time_room_vars = {
            **self.suspect_time_room_vars,
            "Mr. Boddy": self.mr_boddy_time_room_vars,
        }

        # Variables for piece locations at each time
        self.piece_time_room_vars = {
            **self.suspect_time_room_vars,
            **self.weapon_time_room_vars,
            "Mr. Boddy": self.mr_boddy_time_room_vars,
        }
        self.piece_room_time_vars = {
            piece: {
                room: {
                    time: self.piece_time_room_vars[piece][time][room]
                    for time in scenario.times
                }
                for room in scenario.rooms
            }
            for piece in self.piece_time_room_vars
        }

        # Create variables for murder solution
        self.murderer_vars = self.categorical_vars(
            scenario.suspects, suffix=" murderer"
        )
        self.murder_weapon_vars = self.categorical_vars(
            scenario.weapons, suffix=" murder weapon"
        )
        self.murder_room_vars = self.categorical_vars(
            scenario.rooms, suffix=" murder room"
        )
        self.murder_time_vars = self.categorical_vars(
            scenario.times, suffix=" murder time"
        )

        self._add_constraints()

    def _add_constraints(self) -> None:
        """Adds all constraints to the model."""
        # Murder constraints
        for room, room_var in self.murder_room_vars.items():
            for time, time_var in self.murder_time_vars.items():
                # If this is the murder room and time, then only one suspect can be in this room at this time
                self.add(
                    sum(
                        self.suspect_time_room_vars[s][time][room]
                        for s in self.scenario.suspects
                    )
                    == 1
                ).only_enforce_if(room_var, time_var)
        for suspect, suspect_var in self.murderer_vars.items():
            for weapon, weapon_var in self.murder_weapon_vars.items():
                for room, room_var in self.murder_room_vars.items():
                    for time, time_var in self.murder_time_vars.items():
                        # If this is the murder suspect, weapon, room, and time, then the suspect and weapon must be in this room at this time
                        self.add(
                            self.suspect_time_room_vars[suspect][time][room]
                            + self.weapon_time_room_vars[weapon][time][room]
                            == 2
                        ).only_enforce_if(suspect_var, weapon_var, room_var, time_var)

        # Mr. Boddy must be in the murder room at and following the murder time
        accumulated_murder_time_vars = []
        for time, time_room_vars in self.mr_boddy_time_room_vars.items():
            accumulated_murder_time_vars.append(self.murder_time_vars[time])
            for room, room_var in time_room_vars.items():
                self.add(room_var == 1).only_enforce_if(
                    self.murder_room_vars[room],
                    self.any(*accumulated_murder_time_vars),
                )

        # Characters can only move to adjacent rooms
        for time_room_vars in self.character_time_room_vars.values():
            prev_room_vars = None
            for room_vars in time_room_vars.values():
                if prev_room_vars:
                    for prev_room, prev_room_var in prev_room_vars.items():
                        for room, room_var in room_vars.items():
                            if prev_room != room:
                                if not any(
                                    self.scenario.room_coords[prev_room][0]
                                    == self.scenario.room_coords[room][0] + dx
                                    and self.scenario.room_coords[prev_room][1]
                                    == self.scenario.room_coords[room][1] + dy
                                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                ):
                                    # If prev_room and room are not orthogonally adjacent, then a move cannot be made
                                    self.add_at_most_one(prev_room_var, room_var)
                prev_room_vars = room_vars

        # Weapons can only move with suspects
        for time_room_vars in self.weapon_time_room_vars.values():
            prev_time = None
            prev_room_vars = None
            for time, room_vars in time_room_vars.items():
                if prev_time is not None and prev_room_vars is not None:
                    for prev_room, prev_room_var in prev_room_vars.items():
                        for room, room_var in room_vars.items():
                            if prev_room != room:
                                # For the weapon to move, at least one suspect has to have made the same move at the same time
                                self.add_at_least_one(
                                    self.all(
                                        self.suspect_time_room_vars[s][prev_time][
                                            prev_room
                                        ],
                                        self.suspect_time_room_vars[s][time][room],
                                    )
                                    for s in self.scenario.suspects
                                ).only_enforce_if(prev_room_var, room_var)
                prev_time = time
                prev_room_vars = room_vars

    def all(self, *bool_vars: cp_model.IntVar) -> cp_model.IntVar:
        """Returns a variable that is true iff all input variables are true."""
        var_set = frozenset(bool_vars)
        if var_set in self._all_vars:
            return self._all_vars[var_set]
        var = self.new_bool_var(" and ".join(str(var) for var in var_set))
        self.add(sum(var_set) == len(var_set)).only_enforce_if(var)
        self.add(sum(var_set) < len(var_set)).only_enforce_if(~var)
        self._all_vars[var_set] = var
        return var

    def any(self, *bool_vars: cp_model.IntVar) -> cp_model.IntVar:
        """Returns a variable that is true iff any input variables are true."""
        var_set = frozenset(bool_vars)
        if var_set in self._any_vars:
            return self._any_vars[var_set]
        var = self.new_bool_var(" or ".join(str(var) for var in var_set))
        self.add(sum(var_set) >= 1).only_enforce_if(var)
        self.add(sum(var_set) < 1).only_enforce_if(~var)
        self._any_vars[var_set] = var
        return var

    def sum(self, *bool_vars: cp_model.IntVar, value: int) -> cp_model.IntVar:
        """Returns a variable that is true iff the sum of input variables equals value."""
        var_set = frozenset(bool_vars)
        if (var_set, value) in self._sum_vars:
            return self._sum_vars[(var_set, value)]
        var = self.new_bool_var("sum(" + ", ".join(str(var) for var in var_set) + ")")
        self.add(sum(var_set) == value).only_enforce_if(var)
        self.add(sum(var_set) != value).only_enforce_if(~var)
        self._sum_vars[(var_set, value)] = var
        return var

    def categorical_vars(
        self,
        values: list[H],
        *,
        prefix: str = "",
        suffix: str = "",
    ) -> dict[H, cp_model.IntVar]:
        """Creates variables for mutually exclusive categories.

        Args:
            values: List of possible values
            prefix: Optional prefix for variable names
            suffix: Optional suffix for variable names

        Returns:
            Dictionary mapping values to their corresponding variables
        """
        vars = {
            value: self.new_bool_var(f"{prefix}{value}{suffix}") for value in values
        }
        self.add_exactly_one(vars.values())
        return vars

    def categorical_any_equal(
        self, lhs: dict[H, cp_model.IntVar], rhs: dict[H, cp_model.IntVar]
    ) -> cp_model.IntVar:
        return self.any(*(self.all(lhs[key], rhs[key]) for key in lhs))

    def categorical_exactly_one_equal(
        self,
        lhs: dict[H, cp_model.IntVar],
        rhs: dict[H, cp_model.IntVar],
    ) -> cp_model.IntVar:
        return self.sum(*(self.all(lhs[key], rhs[key]) for key in lhs), value=1)

    def motive_time_room_var(
        self, motive: Motive, time: Time, room: Room
    ) -> cp_model.IntVar:
        return self._variable_element_time_room_var(
            self.suspect_time_room_vars,
            self.motive_suspect_vars[motive],
            f"motivated by {motive}",
            time,
            room,
            # number=len(self.scenario.motive_suspects[motive]),
        )

    def motive_room_vars(
        self, motive: Motive, time: Time
    ) -> dict[Room, cp_model.IntVar]:
        return {
            room: self.motive_time_room_var(motive, time, room)
            for room in self.scenario.rooms
        }

    def motive_time_vars(
        self, motive: Motive, room: Room
    ) -> dict[Time, cp_model.IntVar]:
        return {
            time: self.motive_time_room_var(motive, time, room)
            for time in self.scenario.times
        }

    def murderer_time_room_var(self, time: Time, room: Room) -> cp_model.IntVar:
        return self._variable_element_time_room_var(
            self.suspect_time_room_vars, self.murderer_vars, "the murderer", time, room
        )

    def murderer_time_vars(self, room: Room) -> dict[Time, cp_model.IntVar]:
        return {
            time: self.murderer_time_room_var(time, room)
            for time in self.scenario.times
        }

    def murderer_room_vars(self, time: Time) -> dict[Room, cp_model.IntVar]:
        return {
            room: self.murderer_time_room_var(time, room)
            for room in self.scenario.rooms
        }

    def murder_weapon_time_room_var(self, time: Time, room: Room) -> cp_model.IntVar:
        return self._variable_element_time_room_var(
            self.weapon_time_room_vars,
            self.murder_weapon_vars,
            "the murder weapon",
            time,
            room,
        )

    def murder_weapon_time_vars(self, room: Room) -> dict[Time, cp_model.IntVar]:
        return {
            time: self.murder_weapon_time_room_var(time, room)
            for time in self.scenario.times
        }

    def murder_weapon_room_vars(self, time: Time) -> dict[Room, cp_model.IntVar]:
        return {
            room: self.murder_weapon_time_room_var(time, room)
            for room in self.scenario.rooms
        }

    def _variable_element_time_room_var(
        self,
        element_time_room_vars: dict[T, dict[Time, dict[Room, cp_model.IntVar]]],
        variable_element_vars: dict[T, cp_model.IntVar],
        variable: str,
        time: Time,
        room: Room,
        number: int | None = None,
    ) -> cp_model.IntVar:
        key = (
            frozenset(element_time_room_vars.keys()),
            frozenset(variable_element_vars.keys()),
            variable,
            time,
            room,
        )
        if key in self._variable_element_time_room_vars:
            return self._variable_element_time_room_vars[key]
        vars = []
        for (
            element,
            time_room_vars,
        ) in element_time_room_vars.items():
            time_room_var = time_room_vars[time][room]
            variable_var = variable_element_vars[element]
            var = self.new_bool_var(
                f"If {element} is {variable}, then they were in the {room} at {time}"
            )
            self.add(var == 1).only_enforce_if(variable_var, time_room_var)
            self.add(var == 0).only_enforce_if(~self.all(variable_var, time_room_var))
            vars.append(var)
        var = self.sum(*vars, value=number) if number is not None else self.any(*vars)
        self._variable_element_time_room_vars[key] = var  # type: ignore
        return var
