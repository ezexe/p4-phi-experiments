"""
Growth rates of binary registers and Zeckendorf registers.

Prints 2^n against F(n+2), the number of n-digit strings with no two
adjacent 1s, and the digits each needs to hold a given number of levels.
"""

from math import ceil, log2


def main():
    fib = [0, 1]
    for _ in range(40):
        fib.append(fib[-1] + fib[-2])

    print(f"{'digits':>6}{'binary 2^n':>12}{'Zeckendorf':>17}{'ratio':>8}")
    for n in (1, 2, 3, 4, 5, 6, 8, 10, 13):
        print(f"{n:>6}{2 ** n:>12}{fib[n + 2]:>17}{fib[n + 2] / 2 ** n:>8.3f}")

    print(f"\n{'levels':>6}{'binary digits':>15}{'Zeckendorf':>17}{'extra':>7}")
    for levels in (2, 8, 16, 55, 144, 256, 1024, 8192):
        b = ceil(log2(levels))
        f = next(n for n in range(60) if fib[n + 2] >= levels)
        print(f"{levels:>6}{b:>15}{f:>17}{f - b:>7}")


if __name__ == "__main__":
    main()
