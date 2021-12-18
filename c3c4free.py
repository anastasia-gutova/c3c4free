import numpy as np
import codecs
import matplotlib.pyplot as plt
import random

def DFS2(graph, marked, n, vert, start, count):
    marked[vert] = True
    if (n == 0):
        marked[vert] = False
        if graph[vert][start] != 0:
            count = count + 1
            return count
        else:
            return count
    for i in range(len(graph)):
        if marked[i] == False and graph[vert][i] != 0:
            count = DFS2(graph, marked, n-1, i, start, count)
    marked[vert] = False
    return count

def haveCycles( graph, n):
    marked = [False] * len(graph)
    count = 0
    for i in range(len(graph)-(n-1)):
        if (DFS2(graph, marked, n-1, i, i, count) > 0):
            return True
        marked[i] = True
    return False

def Prim(G):
    INF = 9999999
    N = len(G)
    selected_node = np.zeros(N)
    no_edge = 0
    selected_node[0] = True
    res = np.zeros((N, N))
    while (no_edge < N - 1):
        minimum = INF
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and G[m][n]):  
                        if minimum > G[m][n]:
                            minimum = G[m][n]
                            a = m
                            b = n
        res[a][b] = G[a][b]
        res[b][a] = G[a][b]
        selected_node[b] = True
        no_edge += 1
    return res

def createNCycle(matrix):
    res = np.zeros((len(matrix), len(matrix)))

    for i in range(len(res) - 1):
        res[i][i+1] = matrix[i][i+1]
        res[i+1][i] = matrix[i+1][i]
    res[0][len(matrix)-1] = matrix[0][len(matrix)-1]
    res[len(matrix)-1][0] = matrix[len(matrix)-1][0]
    return res

def createNCycle2(matrix, startMatrix, startVIndex):
    res = np.copy(matrix)

    while(startVIndex+4+1 <= len(matrix)):
        res[startVIndex + 4][startVIndex] = startMatrix[startVIndex + 4][startVIndex]
        res[startVIndex][startVIndex + 4] = startMatrix[startVIndex][startVIndex + 4]
        startVIndex += 2
    return res

def getWeight(G):
    sum = 0
    for i in range(len(G)):
        for j in range(len(G)):
            if (i < j):
                sum += G[i][j]
    return sum

def getEdgesCount(G):
    sum = 0
    for i in range(len(G)):
        for j in range(len(G)):
            if (i < j and G[i][j] != 0):
                sum += 1
    return sum

def distance(v1, v2):
    return abs(int(v1[0]) - int(v2[0])) + abs(int(v1[1]) - int(v2[1]))

def showGraph(G, data, weight):
    plt.grid()
    for i in range(0, len(data)):
        plt.plot(data[i][0], data[i][1], 'go')
        plt.text(data[i][0], data[i][1], "%d (%d, %d)" % (i+1, data[i][0], data[i][1]))
    for indA, a in enumerate(G):
        for indB, b in enumerate(G[indA]):
            if(G[indA][indB] != 0):
                plt.plot([data[indA][0], data[indB][0]], [data[indA][1], data[indB][1]], '-')
    plt.title("Weight: %d" % (weight))
    plt.show()

def outputResult(n, weight, matrix):
    writefile = codecs.open("3/Chernyshova%d.txt" % (n), 'w', 'utf-8')
    writefile.write("c Вес подграфа = %d\n" % (weight))
    writefile.write("p edge %d %d\n" % (len(matrix), getEdgesCount(matrix)))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (i < j and matrix[i][j] != 0):
                writefile.write("e %d %d\n" % (i + 1, j + 1))
    print("Created file result: weight: %d" % (weight))
    writefile.close()


files = ["Taxicab_64.txt", "Taxicab_128.txt", "Taxicab_512.txt", "Taxicab_2048.txt", "Taxicab_4096.txt"]
for file in files:
    file1 = open(file, "r")
    raw_data = []
    t = 0
    X = []
    Y = []
    while True:
        line = file1.readline()
        if not line:
            break
        if t != 0:
            a = line.strip().split('\t')

            X.append(int(a[0]))
            Y.append(int(a[1]))
            raw_data.append([int(a[0]), int(a[1])])
        else: n = int(line.split('=')[1])
        t = t + 1
    X = np.array(X)
    Y = np.array(Y)

    data = np.array(raw_data)

    d = len(data)/32 + 2

    print("n= %d"% len(data))
    print("d= %d"% d)

    matrix = []
    for r1 in data:
        row = []
        for r2 in data:
            row.append(abs(int(r1[0]) - int(r2[0])) + abs(int(r1[1]) - int(r2[1])))
        matrix.append(row)

    print(np.array(matrix))
    
    minWeight = 99999999999999999999
    min_matrix = None
    
    matrix_copy = np.copy(matrix) * (-1)
    history = []
    comb = True
    prim_res = createNCycle(matrix_copy)
    prim_res = createNCycle2(prim_res, matrix_copy, 0)
    
    prim_res *= -1
    matrix_copy *= -1
    maxweight = 0
    if(not haveCycles(prim_res, 3) and not haveCycles(prim_res, 4)):
        weight = getWeight(prim_res)
        if(weight > maxweight):
            maxweight = weight
            outputResult(n, weight, prim_res)
            showGraph(prim_res, data, weight)
    else:
        break
    
    for a in range(0,10):
        prim_res_copy = np.copy(prim_res)
        matrix_copy_broken = np.copy(matrix)
        a = 0
        error_count = 0
        history = []
        while(a < 1):
            tempArray = np.zeros((len(prim_res_copy), len(prim_res_copy)))
            tempACounter = 0
            for i in range(len(prim_res_copy)):
                for j in range(len(prim_res_copy)):
                    if (prim_res_copy[i][j] == 0 and i > j):
                        tempACounter +=1
                        tempArray[i][j] = matrix_copy_broken[i][j]
            idx = np.argpartition(tempArray, tempArray.size - tempACounter, axis=None)[-tempACounter:]
            result = np.flip(np.column_stack(np.unravel_index(idx, tempArray.shape)))
            

            randomI = result[:,0]
            randomJ = result[:,1]
            randomI = random.sample(range(0, len(prim_res_copy)), len(prim_res_copy))
            randomJ = random.sample(range(0, len(prim_res_copy)), len(prim_res_copy))
            iter = 0
            edges_counter = getEdgesCount(prim_res_copy)
            for ind, i in enumerate(randomI):
                for jnd, j in enumerate(randomJ):
                    if (i != j):
                        iter+=1
                        (maxi, maxj) = (i,j)
                        if (prim_res_copy[maxi][maxj] == 0):
                            prev_v = prim_res_copy[maxi][maxj]
                            
                            if (prim_res_copy[maxi][maxj] != 0 and prim_res_copy[maxj][maxi] != 0):
                                matrix_copy_broken[maxi][maxj] = -1000000000
                                matrix_copy_broken[maxj][maxi] = -1000000000
                            else:   
                                prim_res_copy[maxi][maxj] = matrix_copy[maxi][maxj]
                                prim_res_copy[maxj][maxi] = matrix_copy[maxj][maxi]
                            if (
                                DFS2(prim_res_copy, [False] * len(prim_res_copy), 2, maxi, maxi, 0) > 0 or
                                DFS2(prim_res_copy, [False] * len(prim_res_copy), 2, maxj, maxj, 0) > 0 or
                                DFS2(prim_res_copy, [False] * len(prim_res_copy), 3, maxi, maxi, 0) > 0 or 
                                DFS2(prim_res_copy, [False] * len(prim_res_copy), 3, maxj, maxj, 0) > 0
                                ):
                                prim_res_copy[maxi][maxj] = prev_v
                                prim_res_copy[maxj][maxi] = prev_v
                                a -= 1
                                error_count +=1
                                if (error_count > 100000000):
                                    break
                            else:
                                error_count = 0
                                edges_counter += 1
                                weight = getWeight(prim_res_copy)
                                if(weight > maxweight):
                                    maxweight = weight
                                    outputResult(n, weight, prim_res_copy)
                        a +=1
        if(not haveCycles(prim_res_copy, 3) and not haveCycles(prim_res_copy, 4)):
            weight = getWeight(prim_res_copy)
            if(weight > maxweight):
                maxweight = weight
                outputResult(n, weight, prim_res_copy)
            
    