from string import ascii_lowercase

def get_square(space):
  r, c = space
  return '{letter}{number}'.format(letter=ascii_lowercase[c], number=r+1)

def get_space(square):
  letter, number = square
  return (
     int(number) - 1,
     ascii_lowercase.index(letter)
  )

