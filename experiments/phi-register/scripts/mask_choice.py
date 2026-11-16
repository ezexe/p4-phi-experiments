"""
How many masks each mask pattern allows.

Counts the masks a pattern allows on a 64-channel layer and prints log₂ of
that count per channel: by Kaplansky's lemma, C(n - k + 1, k) masks keep k of
n channels with no two adjacent, and keeping 2 or 1 of every 4 channels allows
6 or 4 masks per block of 4.
"""

from math import comb, log2

N = 64


def main():
    print(f"{'mask':<28}{'density':>8}{'bits/channel':>14}")
    for density in (0.25, 0.281, 0.375, 0.5):
        k = round(N * density)
        print(f"{'phi':<28}{k / N:>8.3f}{log2(comb(N - k + 1, k)) / N:>14.3f}")
    print(f"{'2 of every 4':<28}{0.5:>8.3f}{log2(6 ** (N // 4)) / N:>14.3f}")
    print(f"{'1 of every 4':<28}{0.25:>8.3f}{log2(4 ** (N // 4)) / N:>14.3f}")
    print(f"{'any 50%, no pattern':<28}{0.5:>8.3f}{log2(comb(N, N // 2)) / N:>14.3f}")


if __name__ == "__main__":
    main()
