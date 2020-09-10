'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''
def DFS(cost, start_point, goals):
    visited=set()
    stack=[start_point]
    path=list()
    while stack:
        ver=stack.pop()
        if ver not in visited:
            visited.add(ver)
            path.append(ver)
            if ver in goals:
                return path
            for i in range(len(cost[ver])-1,0,-1):
                
                if cost[ver][i]>0 and (i not in visited):
                    stack.append(i)
                    
    return []
def DFS(cost, start_point, goals):
    #Add code for DFS
    return []
def UCS(cost, heuristic, goals):
    #Add code for UCS
    return []
def A_Star(cost, heuristic, start_point, goals):
    #Add code for A_Star
    return []
def tri_traversal(cost, heuristic, start_point, goals):
    l = []


    # t1 <= DFS_Traversal
    # t2 <= UCS_Traversal
    # t3 <= A_star_Traversal
    t1=list()
    t2=list()
    t3=list()
    t1=DFS(cost, start_point, goals)
    t2=UCS(cost, start_point, goals)
    t3=A_Star(cost, heuristic, start_point, goals)
    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l