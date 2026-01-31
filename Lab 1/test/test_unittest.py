import unittest
from src.calculator import fun1, fun2, fun3, fun4

class TestCalculator(unittest.TestCase):

    def test_fun1(self):
        self.assertEqual(fun1(1, 2), 3)
        self.assertEqual(fun1(-1, 1), 0)

    def test_fun2(self):
        self.assertEqual(fun2(5, 2), 3)
        self.assertEqual(fun2(2, 5), -3)

    def test_fun3(self):
        self.assertEqual(fun3(3, 4), 12)
        self.assertEqual(fun3(-2, 3), -6)

    def test_fun4(self):
        # fun1(2,3)=5, fun2(2,3)=-1, fun3(2,3)=6. Sum = 10
        self.assertEqual(fun4(2, 3), 10)

if __name__ == '__main__':
    unittest.main()
