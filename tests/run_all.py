"""Run all unittests in the tests folder"""
from unittest import TestLoader, TestResult
from pathlib import Path
from xrd_simulator.utils import _HiddenPrints

def run_tests():
    test_loader = TestLoader()
    test_result = TestResult()

    test_directory = str(Path(__file__).resolve().parent)

    test_suite = test_loader.discover(test_directory, pattern='test_*.py')

    with _HiddenPrints(): # Hide progress bars
        test_suite.run(result=test_result)

    print("\n")
    if test_result.wasSuccessful():
        print(" All unittests ran successfully !")
        exit(0)
    else:
        print(" Unittests Failed !")
        print(test_result.failures)
        print(test_result.errors)
        exit(-1)

if __name__ == '__main__':
    run_tests() # Runs all unittests