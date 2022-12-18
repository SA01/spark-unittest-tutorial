import unittest

from src.addition import add


class TestAddition(unittest.TestCase):
    def test_add(self):
        assert add(1, 2) == 3

    def test_add_alt(self):
        assert add(1, 4) == 5
