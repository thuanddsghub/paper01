import pytest
from calculator import divide

def test_divide_normal_numbers():
assert divide(10,2) == 5

def test_divide_by_zero():
with pytest.raises(ValueError, match-"b must not be zero"):
divide(10,0)
