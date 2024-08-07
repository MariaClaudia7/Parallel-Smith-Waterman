from asyncio import tasks
from re import Match
from venv import logger
import numpy as np
import time
import logging
from Bio import SeqIO
from multiprocessing import Pool, cpu_count, pool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Define scoring function
def pairwise_score(n, m, match, mismatch):
    """
    Define the score for aligning two characters
    """
    if n == m:
        return match
    else:
        return mismatch
    
# Compute cells
def compute_cells(args):
    """
    Compute the score for a specific cell in the scoring matrix
    """
    i, j, SeqA, SeqB, match, mismatch, gap, H = args
    similarity = pairwise_score(SeqA[i - 1], SeqB[j - 1], match, mismatch)
    match_score = H[i - 1, j - 1] + similarity
    delete = H[i - 1, j] + gap
    insert = H[i, j - 1] + gap
    H[i, j] = max(0, match_score, delete, insert)
    return H[i, j], i, j

# Traceback function
def traceback(H, SeqA, SeqB, match, mismatch, gap):
    """
    Traceback through the scoring matrix to find the optimal local alignment
    """
    i, j, = np.unravel_index(np.argmax(H), H.shape)
    align1, align2 = [], []

    while H[i, j] != 0:
        if H[i, j] == H[i - 1, j - 1] + pairwise_score(SeqA[i - 1], SeqB[j - 1], match, mismatch):
            align1.append(SeqA[i - 1])
            align2.append(SeqB[j - 1])
            i -= 1
            j -= 1
        elif H[i, j] == H[i - 1, j] + gap:
            align1.append(SeqA[i - 1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(SeqB[j - 1])
            j -= 1

    score = np.max(H)
    return ''.join(align1[::-1]), ''.join(align2[::-1]), score

# Parallel Smith-Waterman alignment function
def smith_waterman_parallel(SeqA, SeqB, match, mismatch, gap, num_processes):
    """
    Perform the Smith-Watenman local alignment usign parallel processing
    """
    m, n = len(SeqA), len(SeqB)
    H = np.zeros((m + 1, n + 1))

    pool = Pool(num_processes)

    for k in range(1, m + n + 1):
        tasks = []
        for i in range(1, m + 1):
            j = k - i + 1
            if j >= 1 and j <= n:
                tasks.append((i, j, SeqA, SeqB, match, mismatch, gap, H))

            results = pool.map(compute_cells, tasks)
            for value, i, j in results:
                H[i, j] = value

    pool.close()
    pool.join()

    align1, align2, score = traceback(H, SeqA, SeqB, match, mismatch, gap)
    return align1, align2, score

# Main function to execute the entire process
def main():
    try:
        print("welcome to the Parallel Smith-Waterman Alignment Tool")

        SeqA = input("Enter the first sequence: ").strip()
        SeqB = input("Enter the second sequence: ").strip()

        match = int(input("Enter the match score (positive integer): "))
        mismatch = int(input("Enter the mismatch penalty (negative integer): "))
        gap = int(input("Enter the gap penalty (negative integer): "))
        num_processes = int(input(f"Enter the number of processes (1-{cpu_count()}): "))

        start_time = time.time()
        align1, align2, score = smith_waterman_parallel(SeqA, SeqB, match, mismatch, gap, num_processes)
        end_time = time.time()

        print("Alignment 1:\n", align1)
        print("Alignment 2:\n", align2)
        print("Score:", score)
        print(f"Time taken: {end_time - start_time} seconds")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

# Execute the main funtion
if __name__ == '__main__':
    main()