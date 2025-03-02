import black
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, TypedDict
from .types import Answer

if TYPE_CHECKING:
    from .scenario import TemporalClueScenario


prompt_template = """
On a dark winter night, wealthy and enigmatic Mr. John Q. Boddy hosted a small, but lavish, dinner party for some of his closest associates. However, the night ended in tragedy when Mr. Boddy was found dead in one of the rooms of Tudor Mansion in the early hours of the morning. The following persons of interest have been identified as suspects:

{suspects}

And the following weapons were found on the premises:

{weapons}

The murder could only have occured in one of the following rooms:

{rooms}

The rooms are laid out as follows:

{board}

The exact time of the murder is a bit uncertain, but it has been narrowed down to one of the following times:

{times}

At every time the suspects and Mr. Boddy either stayed in their current room or moved to an orthogonally adjacent room (north, south, east, or west). Weapons could be moved by suspects between rooms as well.

Each suspect {uniquely}had one of the following possible motives for killing Mr. Boddy:

{motives}

For the murder to occur, the murderer and Mr. Boddy must have been alone in a room with at least one weapon at some point in the night. Any clue about Mr. Boddy's whereabouts should be read as "Mr. Boddy (dead or alive) ..."

The available clues are as follows:

{clues}

Please answer the following question(s):

{questions}

And the following bonus question(s):

{bonus_questions}

Fill out your final answers in the following format:

{format}

Best of luck, detective.
""".strip()


class TemporalCluePuzzleJsonData(TypedDict):
    num_clues: int
    prompt: str
    solution: dict[str, str]


@dataclass
class TemporalCluePuzzle:
    scenario: "TemporalClueScenario"
    clues: list[str]
    questions: dict[str, Answer]
    bonus_questions: dict[str, Answer]
    tag_counts: dict[str, int]

    @property
    def all_questions(self) -> dict[str, Answer]:
        return {**self.questions, **self.bonus_questions}

    def answer_type(self, answer: Answer) -> str:
        if answer in self.scenario.suspects:
            return "SUSPECT"
        elif answer in self.scenario.weapons:
            return "WEAPON"
        elif answer in self.scenario.rooms:
            return "ROOM"
        elif answer in self.scenario.times:
            return "TIME"
        elif answer in self.scenario.motives:
            return "MOTIVE"
        else:
            return "ANSWER"

    def prompt(self) -> str:
        prompt = prompt_template.format(
            suspects="\n".join(f"• {suspect}" for suspect in self.scenario.suspects),
            weapons="\n".join(f"• {weapon}" for weapon in self.scenario.weapons),
            rooms="\n".join(
                f"{i:0{1 if self.scenario.num_rooms < 10 else 2}d}. {room}"
                for i, room in enumerate(self.scenario.rooms, start=1)
            ),
            board=self.scenario.formatted_board(),
            times="\n".join(f"• {time}" for time in self.scenario.times),
            uniquely=("uniquely " if self.scenario.unique_motives else ""),
            motives="\n".join(
                f"• {motive}"
                + (
                    ""
                    if self.scenario.unique_motives
                    else f" ({len(suspects)} suspect{'' if len(suspects) == 1 else 's'})"
                )
                for motive, suspects in self.scenario.motive_suspects.items()
            ),
            clues="\n".join(f"- {clue}" for clue in self.clues),
            questions="\n".join(
                [
                    f"{chr(65+i)}. {question}"
                    for i, question in enumerate(self.questions.keys())
                ]
            ),
            bonus_questions="\n".join(
                [
                    f"{chr(65+i)}. {question}"
                    for i, question in enumerate(
                        self.bonus_questions.keys(), start=len(self.questions)
                    )
                ]
            ),
            format="\n".join(
                [
                    f"{chr(65+i)}. {self.answer_type(answer)}"
                    for i, answer in enumerate(self.all_questions.values())
                ]
            ),
        )
        if len(self.scenario.times) < 2:
            prompt = re.sub(
                r"\nThe exact time of the murder.*by suspects between rooms as well.\n",
                "",
                prompt,
                flags=re.DOTALL,
            )
            prompt = re.sub(
                r" at some point in the night\. Any clue about Mr\. Boddy's whereabouts should be read as \"Mr\. Boddy \(dead or alive\) \.\.\.\"",
                ".",
                prompt,
                flags=re.DOTALL,
            )
            prompt = re.sub(
                " at least once",
                "",
                prompt,
                flags=re.DOTALL,
            )
            prompt = re.sub(
                " at 12:00 AM",
                "",
                prompt,
                flags=re.DOTALL,
            )
        if len(self.scenario.motives) < 2:
            prompt = re.sub(
                r"Each suspect had one of the following possible motives for killing Mr\. Boddy:.*For the murder to occur",
                "For the murder to occur",
                prompt,
                flags=re.DOTALL,
            )
        return prompt

    def json_data(self) -> "TemporalCluePuzzleJsonData":
        return {
            "num_clues": len(self.clues),
            "prompt": self.prompt(),
            "solution": {
                f"{chr(65+i)}": answer
                for i, answer in enumerate(self.all_questions.values())
            },
        }


TemporalCluePuzzle__repr__ = TemporalCluePuzzle.__repr__


def TemporalCluePuzzle__repr__hook(self) -> str:
    return black.format_str(TemporalCluePuzzle__repr__(self), mode=black.Mode())


TemporalCluePuzzle.__repr__ = TemporalCluePuzzle__repr__hook
