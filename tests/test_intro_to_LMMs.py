import unittest
from intro_to_LMMs import __version__, greet

class TestIntroToLMMs(unittest.TestCase):
    def test_version(self):
        """Test that the version is correctly set."""
        self.assertEqual(__version__, "0.1.0")

    def test_greet(self):
        """Test the greet function."""
        self.assertEqual(greet(), "Welcome to the intro_to_LMMs package!")

if __name__ == "__main__":
    unittest.main()