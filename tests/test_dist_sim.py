# import cupy
# import cupyx
# import numpy as np
# from scipy import sparse
# from scipy.spatial import distance

# from made_recsys.cuda import COSINE_DISTANCE_BETWEEN_ITEMS


# def test_cosine_sim():
#     gen = np.random.default_rng(12123)
#     random_vectors = gen.integers(0, 2, size=(5, 10))
#     true_dist = distance.squareform(distance.pdist(random_vectors, metric="cosine"))
#     true_dist = np.concatenate([row[: i + 1] for i, row in enumerate(true_dist)], axis=0)

#     matrix = sparse.csr_matrix(random_vectors, dtype=np.float32).T

#     half_matrix_size = (matrix.shape[1] * (matrix.shape[1] + 1)) // 2

#     cpu_matrix = cupyx.zeros_pinned((half_matrix_size), dtype=cupy.float32)
#     cuda_matrix = cupy.asarray(cpu_matrix)

#     COSINE_DISTANCE_BETWEEN_ITEMS((1, 1), (matrix.shape[1], matrix.shape[1]),
#                                   (
#         cupy.uint(matrix.shape[1]),
#         cupy.arange(matrix.shape[1], dtype=cupy.uint),
#         cupy.uint(0),
#         cupy.asarray(matrix.indptr, dtype=cupy.uint),
#         cupy.asarray(matrix.indices, dtype=cupy.uint),
#         cupy.asarray(matrix.data, dtype=cupy.float32),
#         cuda_matrix
#     ))

#     cuda_matrix.get(out=cpu_matrix)

#     assert np.allclose(true_dist, cpu_matrix)
