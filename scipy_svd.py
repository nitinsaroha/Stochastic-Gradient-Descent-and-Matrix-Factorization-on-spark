from __future__ import print_function
from pyspark import SparkContext
import sys
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import numpy as np

def CSV_to_sparse(netflix_file):
    # create different lists 
    row_indices = []
    col_indices = []
    data_rating = []

    lines = netflix_file.collect()
    for line in lines:
        line_array = line.split(",")
        row_indices.append(int(line_array[0]) - 1)
        col_indices.append(int(line_array[1]) - 1)
        data_rating.append(float(line_array[2]))
    return csr_matrix((data_rating, (row_indices, col_indices)))

if __name__ == "__main__":
    # made the spark contest
    sc = SparkContext(appName="SVD Solver for Netflix Data")
    # input file
    netflix_file = sc.textFile("nf_subsample.csv")
    sparse_data = CSV_to_sparse(netflix_file)
    # k = 20 principal components
    U, s, Vt = svds(sparse_data, 20)
    # 20, to 20 * 20 to get reconstruction error
    final_s = np.diag(s)

    matrix_after_svd = U.dot((final_s.dot(Vt)))
    nz_index = sparse_data.nonzero()
    # original minus the reconstructed one
    difference = np.asarray(sparse_data[nz_index] - matrix_after_svd[nz_index])
    # recconstruction error
    loss_l2 = np.sum(difference ** 2)
    print(loss_l2)