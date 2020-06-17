import numpy as np

# using array with list
arr1 = np.array([[1, 2, 3], [4, 5, 6]], dtype='float')
print(arr1)

# using array with tuple
arr2 = np.array((1, 2, 3), dtype='float')
print(arr2)

# using zeros
arr3 = np.zeros((2, 3), dtype='float')
print(arr3)

# using arange
arr4 = np.arange(0, 10, 2)
print(arr4)

# using linspace
arr5 = np.linspace(0, 20, 10)
print(arr5)

# using arange and reshape
arr6 = np.arange(15).reshape(3, 5)
print(arr6)

# Create an array having consecutive natural numbers of size 6x3 and print it
qn1_ans = np.arange(18).reshape(6, 3)
print(qn1_ans)