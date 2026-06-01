import hashlib
import time
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


DATA_PATH = "data/ratings.dat"


class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size

    def _hashes(self, item):
        item = str(item)
        for i in range(self.num_hashes):
            hash_value = hashlib.md5((item + str(i)).encode()).hexdigest()
            yield int(hash_value, 16) % self.size

    def add(self, item):
        for h in self._hashes(item):
            self.bit_array[h] = 1

    def contains(self, item):
        return all(self.bit_array[h] == 1 for h in self._hashes(item))

    def memory_usage(self):
        return sys.getsizeof(self.bit_array) + sum(sys.getsizeof(x) for x in self.bit_array)


class CountMinSketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]

    def _hashes(self, item):
        item = str(item)
        for i in range(self.depth):
            hash_value = hashlib.md5((item + str(i)).encode()).hexdigest()
            yield int(hash_value, 16) % self.width

    def add(self, item):
        for row, col in enumerate(self._hashes(item)):
            self.table[row][col] += 1

    def estimate(self, item):
        return min(self.table[row][col] for row, col in enumerate(self._hashes(item)))

    def memory_usage(self):
        total = sys.getsizeof(self.table)
        for row in self.table:
            total += sys.getsizeof(row)
            total += sum(sys.getsizeof(x) for x in row)
        return total


def stream_movielens_ratings(file_path):
    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) == 4:
                user_id, movie_id, rating, timestamp = parts
                yield movie_id


def run_bloom_experiment(file_path, size, num_hashes):
    bloom = BloomFilter(size=size, num_hashes=num_hashes)
    seen = set()

    false_positive = 0
    true_negative = 0
    total = 0

    start_time = time.time()

    for movie_id in stream_movielens_ratings(file_path):
        bloom_result = bloom.contains(movie_id)
        actual_result = movie_id in seen

        if bloom_result and not actual_result:
            false_positive += 1

        if not bloom_result and not actual_result:
            true_negative += 1

        bloom.add(movie_id)
        seen.add(movie_id)
        total += 1

    elapsed = time.time() - start_time

    fpr = false_positive / (false_positive + true_negative)

    return {
        "algorithm": "Bloom Filter",
        "bit_array_size": size,
        "num_hashes": num_hashes,
        "records": total,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_positive_rate": fpr,
        "memory_bytes": bloom.memory_usage(),
        "time_sec": elapsed,
        "throughput_records_per_sec": total / elapsed
    }


def run_cms_experiment(file_path, width, depth):
    cms = CountMinSketch(width=width, depth=depth)
    exact_count = defaultdict(int)

    total = 0
    start_time = time.time()

    for movie_id in stream_movielens_ratings(file_path):
        cms.add(movie_id)
        exact_count[movie_id] += 1
        total += 1

    elapsed = time.time() - start_time

    relative_errors = []
    absolute_errors = []

    for movie_id, true_count in exact_count.items():
        estimated_count = cms.estimate(movie_id)
        abs_error = estimated_count - true_count
        rel_error = abs_error / true_count

        absolute_errors.append(abs_error)
        relative_errors.append(rel_error)

    return {
        "algorithm": "Count-Min Sketch",
        "width": width,
        "depth": depth,
        "records": total,
        "unique_movies": len(exact_count),
        "avg_relative_error": sum(relative_errors) / len(relative_errors),
        "max_absolute_error": max(absolute_errors),
        "memory_bytes": cms.memory_usage(),
        "time_sec": elapsed,
        "throughput_records_per_sec": total / elapsed
    }


def save_graphs(bloom_df, cms_df):
    plt.figure(figsize=(8, 5))
    plt.plot(bloom_df["bit_array_size"], bloom_df["false_positive_rate"], marker="o")
    plt.xlabel("Bit Array Size")
    plt.ylabel("False Positive Rate")
    plt.title("Bloom Filter: Memory Size vs False Positive Rate")
    plt.grid(True)
    plt.savefig("bloom_fpr.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(cms_df["width"], cms_df["avg_relative_error"], marker="o")
    plt.xlabel("Width")
    plt.ylabel("Average Relative Error")
    plt.title("Count-Min Sketch: Width vs Average Relative Error")
    plt.grid(True)
    plt.savefig("cms_error.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(bloom_df["bit_array_size"], bloom_df["memory_bytes"], marker="o", label="Bloom Filter")
    plt.plot(cms_df["width"], cms_df["memory_bytes"], marker="o", label="Count-Min Sketch")
    plt.xlabel("Parameter Size")
    plt.ylabel("Memory Usage Bytes")
    plt.title("Memory Usage Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("memory_comparison.png")
    plt.close()


def main():
    bloom_params = [
        (50000, 3),
        (100000, 5),
        (200000, 7)
    ]

    cms_params = [
        (500, 3),
        (1000, 5),
        (2000, 7)
    ]

    bloom_results = []
    cms_results = []

    print("Running Bloom Filter experiments...")
    for size, num_hashes in bloom_params:
        result = run_bloom_experiment(DATA_PATH, size, num_hashes)
        bloom_results.append(result)
        print(result)

    print("\nRunning Count-Min Sketch experiments...")
    for width, depth in cms_params:
        result = run_cms_experiment(DATA_PATH, width, depth)
        cms_results.append(result)
        print(result)

    bloom_df = pd.DataFrame(bloom_results)
    cms_df = pd.DataFrame(cms_results)

    bloom_df.to_csv("bloom_results.csv", index=False)
    cms_df.to_csv("cms_results.csv", index=False)

    save_graphs(bloom_df, cms_df)

    print("\nBloom Filter Results")
    print(bloom_df)

    print("\nCount-Min Sketch Results")
    print(cms_df)

    print("\nSaved files:")
    print("- bloom_results.csv")
    print("- cms_results.csv")
    print("- bloom_fpr.png")
    print("- cms_error.png")
    print("- memory_comparison.png")


if __name__ == "__main__":
    main()