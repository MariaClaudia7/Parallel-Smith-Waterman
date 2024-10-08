# Parallel Smith-Waterman Alignment Tool

This tool implements a parallelized version of the Smith-Waterman algorithm that is used to find optimal local alignaments
between two sequences. By leveraging multiple processes, it aims to improve the performance of the alignment process on multicore systems.

## Features

- Nucleotide of Amino Acid Sequences
- Customizable Scoring
- Parallel Processing: It takes advantage of multi-core CPUs by using a user-defined number of processes.

## Requirements

- Python 3.x
- NumPy
- Multiprocessing

## Installation

No installation is necessary for running this script if you have Python and the required packages. To install the necessary Python packages, you can use pip:

```bash
pip install numpy multiprocess
```

## Usage

To run the tool, simply execute the script from your command line:

```bash
python Smith_waterman_parallel.py
```

Follow the on-screen prompts to input your sequences and parameters:

1. **First sequence**: The first DNA or protein sequence for alignment.
2. **Second sequence**: The second DNA or protein sequence for alignment.
3. **Match score**: The score for matching characters (positive integer).
4. **Mismatch penalty**: The penalty for mismatching characters (negative integer).
5. **Gap penalty**: The penalty for gaps in alignment (negative integer).
6. **Number of processes**: The number of parallel processes to use (1 to maximum number of CPUs).

## Example

Input:

```plaintext
Enter the first sequence: AGTACGCA
Enter the second sequence: TATGC
Enter the match score (positive integer): 2
Enter the mismatch penalty (negative integer): -1
Enter the gap penalty (negative integer): -1
Enter the number of processes (1-8): 4
```

Output:

```plaintext
Alignment 1: AGTACGCA
Alignment 2: -ATG--C-
Score: 5
Time taken: 0.032 seconds
```

## How it works

### Parallelization strategy

To parallelize the Smith-Waterman algorithm, the tool divides the computation of the scring matrix cells across multiple processes. Here's a brief description of the parallelization approach:

1. Matrix initialization:

    - Initialize a scoring matrix 'H' with dimensions '(len(SeqA) + 1) x (len(SeqB) + 1)', filled with zeros.

2. Anti-diagonal processing:

    - The computation proceeds along the anti-diagonals of the scoring matrix. Cells on the same anti-diagonal are independent of each other and can be computed in parallel.

    - For each anti.diagonal, create a lost of tasks where each task computes the score a specific cell '(i, j)' in the scoing matrix.

3. Parallel Computaion:

    - Use Python's multiprocessing.Pool to distribute the tasks across the available processes.

    - Each process computes the scores for its assigned cells independently.

4. Traceback:

    - After filling the scoring matrix, the traceback procedure is performed to determine the optimal local alignment.

## Key Functions

- pairwise_score(n, m, match, mismatch):
Computes the score for aligning two characters.

- compute_cells(args):
Computes the score for a specific cell in the scoring matrix.

- traceback(H, SeqA, SeqB, match, mismatch, gap):
Traces back through the scoring matrix to find the optimal local alignment.

- smith_waterman_parallel(SeqA, SeqB, match, mismatch, gap, num_processes):
Main function that performs the Smith-Waterman alignment using parallel processing.

## Contributing

Contributions to the project are welcome. You can contribute in several ways:

- Reporting issues
- Adding new features
- Improving documentation
- Refactoring code for better efficiency or readability


