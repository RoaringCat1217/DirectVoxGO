import numpy as np
from tqdm import tqdm


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

    for normal in normals:
        f.write("vn {} {} {}\n".format(*normal))

    for triangle in triangles:
        f.write("f")
        for index in triangle:
            f.write(" {}//{}".format(index + 1, index + 1))
        f.write("\n")
    f.close()


if __name__ == '__main__':
    print("Loading mesh...")
    vertices, colors, normals, triangles, edges = load_mesh('mesh.obj')
    n = len(vertices)
    parent = np.arange(n)
    size = np.ones(n)

    def find_root(i):
        if parent[i] == i:
            return i
        root = find_root(parent[i])
        parent[i] = root
        return root

    def union(i, j):
        i = find_root(i)
        j = find_root(j)
        if i == j:
            return
        if size[i] <= size[j]:
            parent[i] = j
            size[j] += size[i]
            size[i] = 0
        else:
            parent[j] = i
            size[i] += size[j]
            size[j] = 0

    print("Processing edges...")
    for i, j in tqdm(edges):
        union(i, j)

    root = np.argmax(size)
    remapping = {}
    n_new = 0
    vertices_new = []
    colors_new = []
    normals_new = []
    triangles_new = []
    print("Filtering vertices...")
    for i in tqdm(range(n)):
        if find_root(i) == root:
            remapping[i] = n_new
            n_new += 1
            vertices_new.append(vertices[i])
            colors_new.append(colors[i])
            normals_new.append(normals[i])
    print("Filtering triangles...")
    for i, j, k in tqdm(triangles):
        if i in remapping:
            triangles_new.append([remapping[i], remapping[j], remapping[k]])
    vertices_new = np.array(vertices_new, dtype=np.float32)
    colors_new = np.array(colors_new, dtype=np.float32)
    normals_new = np.array(normals_new, dtype=np.float32)
    triangles_new = np.array(triangles_new, dtype=np.int32)

    print("Saving mesh...")
    save_mesh('cleaned.obj', vertices_new, colors_new, normals_new, triangles_new)
