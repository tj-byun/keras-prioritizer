import dsa
a = [[1.0, .2, .3], [2, 3, 1]]
b = [[0.4, .5, .6], [5, 32, 4]]
c = [[0, 1, 0], [1, 0, 0]]
print(a)
print(b)
print(c)

#dsa.hey(b)
result = dsa.get_dist_matrix(a, b, c)
print(result)

