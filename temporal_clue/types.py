from typing import (
    Hashable,
    Literal,
    NewType,
    ParamSpec,
    TypeVar,
    Union,
)

Suspect = NewType("Suspect", str)
Weapon = NewType("Weapon", str)
Room = NewType("Room", str)
Time = NewType("Time", str)
Motive = NewType("Motive", str)
MrBoddy = Literal["Mr. Boddy"]
Character = Union[Suspect, MrBoddy]
Piece = Union[Character, Weapon]
Element = Union[Piece, Room, Time]
Answer = Union[Suspect, Weapon, Room, Time, Motive]

H = TypeVar("H", bound=Hashable)
P = ParamSpec("P")
T = TypeVar("T")
