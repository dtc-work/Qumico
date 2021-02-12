import unittest
import sys


def suite_helper_by_discover(test_dir):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.discover(test_dir))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite_helper_by_discover("./"))
    if not result.wasSuccessful():
        sys.exit(1) # failure or error exist
