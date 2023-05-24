import numpy as np


def load_mesh(path):
    f = open(path, 'r')
    vertices = []
    colors = []
    normals = []
    triangles = []
    edges = []
    while True:
        line = f.readline()
        if not line:
            break
        line = line.split()
        if line[0] == 'v':
            vertex = [float(num) for num in line[1:4]]
            color = [float(num) for num in line[4:7]]
            vertices.append(vertex)
            colors.append(color)
        elif line[0] == 'vn':
            normal = [float(num) for num in line[1:4]]
            normals.append(normal)
        else:
            triangle = [int(entry.split('//')[0]) - 1 for entry in line[1:4]]
            triangles.append(triangle)
            edges.extend([[triangle[0], triangle[1]],
                          [triangle[1], triangle[2]],
                          [triangle[2], triangle[0]]])
    vertices = np.array(vertices, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    triangles = np.array(triangles, dtype=np.int32)
    edges = np.array(edges, dtype=np.int32)
    f.close()
    return vertices, colors, normals, triangles, edges


def save_mesh(path, vertices, colors, normals, triangles):
    f = open(path, 'w')
    for i in range(len(vertices)):
        f.write("v {} {} {}".format(*vertices[i]))
        f.write(" {} {} {}\n".format(*colors[i]))

    if normals:
        for normal in normals:
            f.write("vn {} {} {}\n".format(*normal))

        for triangle in triangles:
            f.write("f")
            for index in triangle:
                f.write(" {}//{}".format(index + 1, index + 1))
            f.write("\n")
    else:
        for triangle in triangles:
            f.write("f")
            for index in triangle:
                f.write(" {}".format(index + 1))
            f.write("\n")
    f.close()
