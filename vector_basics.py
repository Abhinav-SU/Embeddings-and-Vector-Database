# Vectors are arrays of number can be represented using Numpy

import numpy as np 

vector1 = np.array([1,0])
vector2 = np.array([0,1])
print(vector1)
print(type(vector1))
print(vector2)
print(type(vector2))

# vector Dimension, Vector Magnitude, Dot Product
'''
Vector dimenions is number of elements it contain
Vector Magnitude represent vetor size or length
DOt product is number given by  u.v =|u||v|cos(theta) also used to show similarity between two vectors
orthogonal vector if there dot product is zero

'''

v1 =np.array([1,0])
v2 =np.array([0,1])
v3 =np.array([np.sqrt(2),np.sqrt(2)])

#Dimensions
print(v1.shape)

#Magnitude
print(np.sqrt(np.sum(v1**2)))

print(np.linalg.norm(v1))
print(np.linalg.norm(v3))

#Dot Product
print(np.sum(v1*v2))
print(v1@v3)
