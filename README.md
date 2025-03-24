![image](https://github.com/user-attachments/assets/d36443e9-197b-4413-b045-09690b11ade3)


# Temporal Clue

A benchmark testing LLMs' deductive reasoning through Clue-inspired puzzles with temporal and spatial dimensions. The puzzles are generated with code and clues chosen using OR-Tools' CP-SAT solver. See [example.ipynb](https://github.com/bradhilton/temporal-clue/blob/main/example.ipynb) for how to generate your own puzzles.

## Overview

Tests an LLM's ability to:

- Track entities across time and space
- Apply logical constraints and conditionals
- Integrate information across clues

## Format

Each puzzle provides:

- Suspects
- Weapons
- Rooms with defined spatial relationships
- Times (optional)
- Motives (optional)
- Logical clues

LLMs must then answer a series of questions, usually including the classic who, (with) what, and where, and the new when and why, as well as a few bonus questions. Here is a simple example with few variables and clues:

```md
On a dark winter night, wealthy and enigmatic Mr. John Q. Boddy hosted a small, but lavish, dinner party for some of his closest associates. However, the night ended in tragedy when Mr. Boddy was found dead in one of the rooms of Tudor Mansion in the early hours of the morning. The following persons of interest have been identified as suspects:

• Professor Plum
• Colonel Mustard

And the following weapons were found on the premises:

• Poison
• Horseshoe

The murder could only have occured in one of the following rooms:

1. Fountain
2. Ballroom

The rooms are laid out as follows:

  N  
W 1 E
W 2 E
  S

The exact time of the murder is a bit uncertain, but it has been narrowed down to one of the following times:

• 11:45 PM
• 12:00 AM

At every time the suspects and Mr. Boddy either stayed in their current room or moved to an orthogonally adjacent room (north, south, east, or west). Weapons could be moved by suspects between rooms as well.

Each suspect uniquely had one of the following possible motives for killing Mr. Boddy:

• Jealousy
• Hatred

For the murder to occur, the murderer and Mr. Boddy must have been alone in a room with at least one weapon at some point in the night. Any clue about Mr. Boddy's whereabouts should be read as "Mr. Boddy (dead or alive) ..."

The available clues are as follows:

- Professor Plum was in the same room as Colonel Mustard at 12:00 AM
- The Horseshoe was in the room just south of Mr. Boddy at 11:45 PM
- Colonel Mustard was in the Ballroom at 11:45 PM or Professor Plum was at the Fountain at 12:00 AM
- Colonel Mustard and the Horseshoe were in the Ballroom together at least once
- The Horseshoe was in the room just south of the Poison at 12:00 AM
- Colonel Mustard is motivated by Hatred

Please answer the following question(s):

A. Who murdered Mr. Boddy?
B. What weapon did the murderer use?
C. Where was the murder committed?
D. When did the murder occur?
E. Why did the murderer do it?

And the following bonus question(s):

F. Where was the suspect motivated by Jealousy at 12:00 AM?
G. Where was Professor Plum at 11:45 PM?

Fill out your final answers in the following format:

A. SUSPECT
B. WEAPON
C. ROOM
D. TIME
E. MOTIVE
F. ROOM
G. ROOM

Best of luck, detective.
```

The solution to the above puzzle is:

```md
A. Professor Plum
B. Poison
C. Fountain
D. 11:45 PM
E. Jealousy
F. Ballroom
G. Fountain
```
