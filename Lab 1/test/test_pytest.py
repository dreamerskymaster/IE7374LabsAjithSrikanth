import pytest
from src.calculator import fun1, fun2, fun3, fun4

def test_fun1():
    assert fun1(1, 2) == 3
    assert fun1(-1, 1) == 0

def test_fun2():
    assert fun2(5, 2) == 3
    assert fun2(2, 5) == -3

def test_fun3():
    assert fun3(3, 4) == 12
    assert fun3(-2, 3) == -6

def test_fun4():
    # fun1(2,3)=5, fun2(2,3)=-1, fun3(2,3)=6. Sum = 10
    assert fun4(2, 3) == 10
