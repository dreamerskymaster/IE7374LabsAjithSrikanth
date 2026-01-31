def fun1(x, y):
    """Adds two input numbers, x and y."""
    return x + y

def fun2(x, y):
    """Subtracts y from x."""
    return x - y

def fun3(x, y):
    """Multiplies x and y."""
    return x * y

def fun4(x, y):
    """Combines the results of fun1, fun2, and fun3 and returns their sum."""
    return fun1(x, y) + fun2(x, y) + fun3(x, y)

def fun_power(x, y):
    """Returns x raised to the power of y."""
    return x ** y

def fun_sqrt(x):
    """Returns the square root of x. Raises ValueError for negative numbers."""
    if x < 0:
        raise ValueError("Cannot take square root of negative number")
    return x ** 0.5

def fun_avg(numbers):
    """Returns the average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
