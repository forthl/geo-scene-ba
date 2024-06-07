import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_algorithm(cost_matrix):
    """
    input:
    cost_matrix: (n, n)

    output:
    assignments: (n,) ith element is column index assigned to ith row
    min_cost: minimum cost
    """

    # solve assignment problem
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    assignments = np.zeros(cost_matrix.shape[0], dtype=int)
    assignments[row_idx] = col_idx

    # compute minimum cost
    min_cost = cost_matrix[row_idx, col_idx].sum()

    return assignments, min_cost