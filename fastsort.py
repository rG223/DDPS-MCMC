import numpy as np

def compare(p1, p2):
    # return 0同层 1 p1支配p2  -1 p2支配p1
    D = len(p1)
    p1_dominate_p2 = True  # p1 更小
    p2_dominate_p1 = True
    for i in range(D):
        if p1[i] > p2[i]:
            p1_dominate_p2 = False
        if p1[i] < p2[i]:
            p2_dominate_p1 = False

    if p1_dominate_p2 == p2_dominate_p1:
        return 0
    return 1 if p1_dominate_p2 else -1

def fast_non_dominated_sort(P):
    P_size = len(P)
    n = np.full(shape=P_size, fill_value=0)    # 被支配数
    S = []    # 支配的成员
    f = []    # 0 开始每层包含的成员编号们
    rank = np.full(shape=P_size, fill_value=-1)  # 所处等级

    f_0 = []
    for p in range(P_size):
        n_p = 0
        S_p = []
        for q in range(P_size):
            if p == q:
                continue
            cmp = compare(P[p], P[q])
            if cmp == 1:
                S_p.append(q)
            elif cmp == -1:  # 被支配
                n_p += 1
        S.append(S_p)
        n[p] = n_p
        if n_p == 0:
            rank[p] = 0
            f_0.append(p)
    f.append(f_0)

    i = 0
    while len(f[i]) != 0:  # 可能还有i+1层
        Q = []
        for p in f[i]:  # i层中每个个体
            for q in S[p]:  # 被p支配的个体
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        f.append(Q)
    rank +=1
    return rank,f

if __name__ == '__main__':
    P = np.array([[2,2,5],[2,1,3],[1,2,3],[3,1,4],[2,1,4],[1,3,5],[1,3,3]])
    rank, f = fast_non_dominated_sort(P)
    f.pop()         # 去掉最后的空集
    print(rank)
    print(f)

