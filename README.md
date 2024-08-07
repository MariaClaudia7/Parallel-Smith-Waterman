# Parallel Smith-Waterman Alignment Tool

This tool implements a parallelized version of the Smith-Waterman algorithm for local sequence alignment using Python. By leveraging multiple processes, it aims to improve the performance of the alignment process on multicore systems.

## Features

- Parallel computation of Smith-Waterman scoring matrix.
- Traceback to extract the optimal local alignment.
- User inputs for sequences and scoring parameters.
- Performance metrics including computation time.

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

## Contributing

Contributions to the project are welcome. You can contribute in several ways:

- Reporting issues
- Adding new features
- Improving documentation
- Refactoring code for better efficiency or readability


