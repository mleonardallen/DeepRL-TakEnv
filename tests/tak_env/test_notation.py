import pytest
from pytest import fixture
from tak_env import board, types, notation

def describe_get_square():

  def should_convert_space_to_square():
    assert notation.get_square((2,0), 3) == 'a1'
    assert notation.get_square((1,0), 3) == 'a2'
    assert notation.get_square((0,0), 3) == 'a3'
    assert notation.get_square((2,1), 3) == 'b1'
    assert notation.get_square((1,1), 3) == 'b2'
    assert notation.get_square((0,1), 3) == 'b3'

def describe_get_space():

  def should_convert_sqare_to_space():
    assert notation.get_space('a1', 3) == (2,0)
    assert notation.get_space('a2', 3) == (1,0)
    assert notation.get_space('a3', 3) == (0,0)
    assert notation.get_space('b1', 3) == (2,1)
    assert notation.get_space('b2', 3) == (1,1)
    assert notation.get_space('b3', 3) == (0,1)

def describe_get_action():
  testdata = [
    ('a6', 6, {'action': 'place', 'to': (0,0), 'piece': 1}),
    ('Fc5', 6, {'action': 'place', 'to': (1,2), 'piece': 1}),
    ('Sd2', 6, {'action': 'place', 'to': (4,3), 'piece': 2}),
    ('Ce4', 6, {'action': 'place', 'to': (2,4), 'piece': 3}),
    ('1e4<1', 6, {'action': 'move', 'carry': (1,), 'direction': '<', 'from': (2,4)}),
    ('1d3<1', 6, {'action': 'move', 'carry': (1,), 'direction': '<', 'from': (3,3)}),
    ('1d1+1', 6, {'action': 'move', 'carry': (1,), 'direction': '+', 'from': (5,3)}),
    ('a1>', 6, {'action': 'move', 'carry': (1,), 'direction': '>', 'from': (5,0)}),
    ('4c3>', 6, {'action': 'move', 'carry': (4,), 'direction': '>', 'from': (3,2)}),
    ('3b2+111', 6, {'action': 'move', 'carry': (1,1,1,), 'direction': '+', 'from': (4,1)}),
    ('2d4-2C', 6, {'action': 'move', 'carry': (2,), 'direction': '-', 'from': (2,3)}),
    ('5e4<23', 6, {'action': 'move', 'carry': (2,3,), 'direction': '<', 'from': (2,4)}),
  ]

  @pytest.mark.parametrize("ptn,size,expected", testdata)
  def it_does(ptn, size, expected):
      assert notation.to_action(ptn, size) == expected

def describe_standardize():
  testdata = [
    ('a6', 'Fa6'),
    ('Fc5', 'Fc5'),
    ('Sd2', 'Sd2'),
    ('Ce4', 'Ce4'),
    ('1e4<1', '1e4<1'),
    ('1d3<1', '1d3<1'),
    ('1d1+1', '1d1+1'),
    ('a1>', '1a1>1'),
    ('4c3>', '4c3>4'),
    ('3b2+111', '3b2+111'),
    ('2d4-2C', '2d4-2'),
    ('5e4<23', '5e4<23'),
  ]

  @pytest.mark.parametrize("ptn,expected", testdata)
  def should_standardize(ptn, expected):
      assert notation.standardize(ptn) == expected