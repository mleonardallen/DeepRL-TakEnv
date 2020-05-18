from pytest import fixture
from tak_env import board, types, notation

def describe_get_square():

  def should_convert_space_to_square():
    assert notation.get_square((0,0)) == 'a1'
    assert notation.get_square((1,0)) == 'a2'
    assert notation.get_square((0,1)) == 'b1'
    assert notation.get_square((1,1)) == 'b2'

def describe_get_space():

  def should_convert_sqare_to_space():
    assert notation.get_space('a1') == (0,0)
    assert notation.get_space('a2') == (1,0)
    assert notation.get_space('b1') == (0,1)
    assert notation.get_space('b2') == (1,1)
