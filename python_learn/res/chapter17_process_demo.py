from concurrent.futures import ProcessPoolExecutor
import os


def square(num):
    return os.getpid(), num, num * num


if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]

    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(square, numbers))

    for pid, num, result in results:
        print(f"pid={pid}, {num} 的平方是 {result}")
