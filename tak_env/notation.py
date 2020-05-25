from string import ascii_lowercase
from tak_env.types import Direction, Stone, StoneLetter

def get_action(ptn, size):
  ptn = standardize(ptn)
  direction = get_movement_direction(ptn)

  if direction:
    fromSpace, carry = ptn.split(direction)
    return {
      'action': 'move',
      'carry': tuple([int(x) for x in carry]),
      'from': get_space(fromSpace[-2:], size),
      'direction': direction,
    }

  return { 
    'action': 'place',
    'to': get_space(ptn[-2:], size),
    'piece': get_stone_value(ptn[0])
  }

def standardize(ptn):
  direction = get_movement_direction(ptn)
  if direction:
    # omitted first character assumes 1 stone moved
    if not ptn[0].isdigit():
      ptn = '1' + ptn
    # omitted carry assumes total carry
    parts = ptn.split(direction)
    if not parts[1]:
      ptn = ptn + ptn[0]
    if ptn[-1:] == StoneLetter.CAPITAL.value:
      ptn = ptn[:-1]
  else:
    # omitted first character assumes flat stone
    if not any(ptn[0] in x for x in [
      StoneLetter.CAPITAL.value,
      StoneLetter.FLAT.value,
      StoneLetter.STANDING.value,
    ]):
      ptn = StoneLetter.FLAT.value + ptn
  return ptn

def get_stone_value(letter):
  if letter == StoneLetter.FLAT.value:
    return Stone.FLAT.value
  if letter == StoneLetter.STANDING.value:
    return Stone.STANDING.value
  if letter == StoneLetter.CAPITAL.value:
    return Stone.CAPITAL.value
  return None

def get_movement_direction(ptn):
  if Direction.UP.value in ptn:
    return Direction.UP.value
  if Direction.DOWN.value in ptn:
    return Direction.DOWN.value
  if Direction.LEFT.value in ptn:
    return Direction.LEFT.value
  if Direction.RIGHT.value in ptn:
    return Direction.RIGHT.value
  return None

def get_square(space, size):
  r, c = space
  return '{letter}{number}'.format(letter=ascii_lowercase[c], number=size-r)

def get_space(square, size):
  letter, number = square
  return (
     size - int(number),
     ascii_lowercase.index(letter)
  )
