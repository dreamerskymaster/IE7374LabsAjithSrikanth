import unittest
from src.calculator import fun1, fun2, fun3, fun4, fun_power, fun_sqrt, fun_avg

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

    def test_fun_power(self):
        self.assertEqual(fun_power(2, 3), 8)
        self.assertEqual(fun_power(5, 0), 1)

    def test_fun_sqrt(self):
        self.assertEqual(fun_sqrt(16), 4)
        self.assertEqual(fun_sqrt(0), 0)
        with self.assertRaises(ValueError):
            fun_sqrt(-1)

    def test_fun_avg(self):
        self.assertEqual(fun_avg([1, 2, 3, 4, 5]), 3)
        self.assertEqual(fun_avg([]), 0)

if __name__ == '__main__':
    unittest.main()
