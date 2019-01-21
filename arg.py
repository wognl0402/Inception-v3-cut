import numpy as np


def k_largest_index_argsort(a_ori, k, slice_map=None): 
    a = a_ori.copy()
    if slice_map is not None:
        for i in range(slice_map.shape[0]):
            for j in range(slice_map.shape[1]):
                if slice_map[i,j] == 0:
                    a[i,j] = -float("inf")
    idx = np.argsort(a.ravel())[:-k-1:-1] 
    return np.column_stack(np.unravel_index(idx, a.shape)) 

def k_smallest_index_argsort(a_ori, k, slice_map=None):
    a = a_ori.copy()
    if slice_map is not None:
        for i in range(slice_map.shape[0]):
            for j in range(slice_map.shape[1]):
                if slice_map[i,j] == 0:
                    a[i,j] = float("inf")
    idx = np.argsort(a.ravel())[:k] 
    return np.column_stack(np.unravel_index(idx, a.shape)) 


blind = np.ones((5,5))
blind[0,0] = 0
blind[2,2] = 0

test = np.zeros((5,5))
test[0,0] = -1
test[0,1] = -2
test[0,3] = -1
test[2,2] = 1
test[2,0] = 3
test[2,3] = 3
test_max = k_largest_index_argsort(test, 4, blind)
test_min = k_smallest_index_argsort(test,4, blind)
print('test_max', test_max)
print('test_min', test_min)
for i, j in test_min:
	print(i,j)
print(test_min[0][0])
print(test_min[0][1])
#print(test[test_min[0]])
