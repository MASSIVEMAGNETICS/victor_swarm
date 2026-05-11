def is_prime(n: int) -> bool:
    """
    Checks if a number is a prime number.

    A prime number is a positive integer greater than 1 that has no positive divisors other than 1 and itself.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
