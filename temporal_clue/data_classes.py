from dataclasses import dataclass, field
from ortools.sat.python import cp_model
from typing import Callable

from .model import TemporalClueCpModel
from .types import Answer, Character, Room, Suspect, Time, Weapon


@dataclass
class CharacterMove:
    character: Character
    time: Time
    from_room: Room
    to_room: Room


@dataclass
class WeaponMove:
    weapon: Weapon
    suspect: Suspect
    time: Time
    from_room: Room
    to_room: Room


@dataclass
class TemporalClue:
    description: str
    get_bool_var: Callable[[TemporalClueCpModel], cp_model.IntVar]
    bias: float = 0.0
    question_answer: tuple[str, Answer] | None = None
    tags: list[str] = field(default_factory=list)
