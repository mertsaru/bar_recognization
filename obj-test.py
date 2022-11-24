import GraphObj

obj_1 = GraphObj.Graph("dataset_1/A/A-kappa/Z_80074574_0230_IFIX1.jpg")

print(obj_1.name)

matrix = obj_1.to_matrix()
print(matrix)