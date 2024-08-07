from asyncio import tasks
from ctypes import alignment
from msilib import sequence
from re import Match
from venv import logger
import numpy as np
import time
import logging
# from Bio import SeqIO
from multiprocessing import Pool, cpu_count

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Define scoring function
def pairwise_score(n: str, m: str, match: int, mismatch: int) -> int:
    """
    Define the score for aligning two characters
    """
    if n == m:
        return match
    else:
        return mismatch
    
# Compute cells
def compute_cells(args: tuple[int, int, str, str, int, int, int, np.ndarray]) -> tuple[int, int, int]:
    """
    Compute the score for a specific cell in the scoring matrix
    """
    i, j, sequence1, sequence2, match, mismatch, gap, scoring_matrix = args
    similarity = pairwise_score(sequence1[i - 1], sequence2[j - 1], match, mismatch)
    match_score = scoring_matrix[i - 1, j - 1] + similarity
    delete = scoring_matrix[i - 1, j] + gap
    insert = scoring_matrix[i, j - 1] + gap
    scoring_matrix[i, j] = max(0, match_score, delete, insert)
    return scoring_matrix[i, j], i, j

# Traceback function
def traceback(scoring_matrix: np.ndarray, sequence1: str, sequence2: str, match: int, mismatch: int, gap: int) -> tuple[str, str, int]:
    """
    Traceback through the scoring matrix to find the optimal local alignment
    """
    i, j, = len(sequence1) - 1, len(sequence2) - 1
    alignment1, alignment2 = [], []

    while scoring_matrix[i, j] != 0:
        if i > 0 and j > 0 and scoring_matrix[i, j] == scoring_matrix[i - 1, j - 1] + pairwise_score(sequence1[i - 1], sequence2[j - 1], match, mismatch):
            alignment1.append(sequence1[i - 1])
            alignment2.append(sequence2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and scoring_matrix[i, j] == scoring_matrix[i - 1, j] + gap:
            alignment1.append(sequence1[i - 1])
            alignment2.append('-')
            i -= 1
        elif j > 0 and scoring_matrix[i, j] == scoring_matrix[i, j - 1] + gap:
            alignment1.append('-')
            alignment2.append(sequence2[j - 1])
            j -= 1
        else:
            break

    score = np.max(scoring_matrix)
    return ''.join(alignment1[::-1]), ''.join(alignment2[::-1]), score

# Parallel Smith-Waterman alignment function
def smith_waterman_parallel(sequence1: str, sequence2: str, match: str, mismatch: str, gap: int, num_processes: int) -> tuple[str, str, int]:
    """
    Perform the Smith-Watenman local alignment usign parallel processing
    """
    m, n = len(sequence1), len(sequence2)
    scoring_matrix = np.zeros((m + 1, n + 1))

    pool = Pool(num_processes)

    for k in range(1, m + n + 1):
        tasks = []
        for i in range(1, m + 1):
            j = k - i + 1
            if j >= 1 and j <= n:
                tasks.append((i, j, sequence1, sequence2, match, mismatch, gap, scoring_matrix))

            results = pool.map(compute_cells, tasks)
            for value, i, j in results:
                scoring_matrix[i, j] = value

    pool.close()
    pool.join()

    alignment1, alignment2, score = traceback(scoring_matrix, sequence1, sequence2, match, mismatch, gap)
    return alignment1, alignment2, score

# Main function to execute the entire process
def main() -> None:
    """
    Main function to execute the entire parallel Smith-Waterman alignment processes
    Handles user input and displays the alignment results
    """
    try:
        print("Welcome to the Parallel Smith-Waterman Alignment Tool")

        while True:
            sequence1 = input("Enter the first sequence: ").strip().upper()
            try:
                if not sequence1.isalpha():
                    raise ValueError("The sequence contains non-letter characters")
                if not all(c in "ATCGU" for c in sequence1) or not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in sequence1):
                    raise ValueError("Please enter a nucleotide or aminoacid sequence")
                break
            except ValueError as e:
                print(f"Invalid input: {e}")

        while True:
            sequence2 = input("Enter the first sequence: ").strip().upper()
            try:
                if not sequence2.isalpha():
                    raise ValueError("The sequence contains non-letter characters")
                if not all(c in "ATCG" for c in sequence2) or not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in sequence2):
                    raise ValueError("Please enter a nucleotide or aminoacid sequence")
                break
            except ValueError as e:
                print(f"Invalid input: {e}")
        
        while True:
            try:
                match = int(input("Enter the match score (positive integer): "))
                if match > 0:
                    break
                else:
                    print("Invalid input: Please enter a positive integer")
            except ValueError as e:
                print(f"Invalid input: {e}")
                
        while True:
            try:
                mismatch = int(input("Enter the mismatch penalty (negative integer): "))
                if mismatch < 0:
                    break
                else:
                    print("Invalid input: Please enter a negative integer")
            except ValueError as e:
                print(f"Invalid input: {e}")

        while True:
            try:
                gap = int(input("Enter the gap penalty (negative integer): "))
                if gap < 0:
                    break
                else:
                    print("Invalid input: Please enter a negative integer")
            except ValueError as e:
                print(f"Invalid input: {e}")

        
        while True:
            try:
                num_processes = int(input(f"Enter the number of processes (1-{cpu_count()}): "))
                if 1 <= num_processes <= cpu_count():
                    break
                else:
                    print(f"Invalid input: Please enter a number between 1 and {cpu_count()}")
            except ValueError as e:
                print(f"Invalid input: {e}")

        start_time = time.time()
        alignment1, alignment2, score = smith_waterman_parallel(sequence1, sequence2, match, mismatch, gap, num_processes)
        end_time = time.time()

        print("Alignment 1:\n", alignment1)
        print("Alignment 2:\n", alignment2)
        print("Score:", score)
        print(f"Time taken: {end_time - start_time} seconds")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

# Execute the main funtion
if __name__ == '__main__':
    main()