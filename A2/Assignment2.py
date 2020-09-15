from queue import PriorityQueue
def DFS_Traversal(cost, start_point, goals):
    #test whether all goals are valid
    valid=[i for i in goals if i in range(1,len(cost[0]))]
    if len(valid)==0:
        return[]
    visited=set()#Keep track of visited nodes
    stack=[start_point]#stack implementation in python
    paths=dict()#Holds path to each node when computed
    paths[start_point]=[start_point]#Path of initial node is node itself
    #While stack is not empty do
    while stack:
        ver=stack.pop()#pop node from stack
        if ver not in visited:
            visited.add(ver)#if node not in visited add to visited
            #If the popped node is the goal return path to that node
            if ver in goals:
                return paths[ver]
            #explore neighbors of popped node
            for i in range(len(cost[ver])-1,0,-1):
                if cost[ver][i]>0 and (i not in visited):
                    stack.append(i)#if neighbor and not yet visited push to stack
                    paths[i]=paths[ver]+[i]#Add path to neighbor node
    #if path to goal is not found return empty list
    return []

def UCS_Traversal(cost, start_point, goals):
    #test whether atleast 1 goal is valid
    valid=[i for i in goals if i in range(1,len(cost[0]))]
    if len(valid)==0:
        return[]
    visited = set()
    q=PriorityQueue()
    q.put((0, start_point, [start_point]))
    while not q.empty(): 
        cum_cost, curr, path = q.get()
        visited.add(curr)
        if curr in goals:
            return path
        else:
            children=list()
            row_cost = cost[curr]
            for i in range(1, len(row_cost)):
                if row_cost[i]>0:
                    children.append(i)
            for i in children:
                if i not in visited:
                    q.put((cum_cost+cost[curr][i], i, path+[i]))
    return []

def A_star_Traversal(cost, heuristic, start_point, goals):
    #test whether atleast 1 goal is valid
    valid=[i for i in goals if i in range(1,len(cost[0]))]
    if len(valid)==0:
        return[]
    visited = set()#keeps track of all visited nodes
    diction = dict()#dict to store paths
    q=PriorityQueue()
    q.put((heuristic[start_point], start_point, 0,heuristic[start_point], [start_point]))
    while not q.empty():
        cum_cost, curr, g, h, path = q.get()
        visited.add(curr)
        if curr in goals:
            diction[g]=path
        else:
            children=list()
            row_cost=cost[curr]
            for i in range(1,len(row_cost)):
                if row_cost[i]>0:
                    children.append(i)
            for i in children:
                if i not in visited:
                    g_new=g+cost[curr][i]
                    h_new=heuristic[i]
                    f_new=g_new+h_new
                    q.put((f_new, i, g_new, h_new, path + [i]))
    return diction.get(min(diction.keys()))

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
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

def tri_traversal(cost, heuristic, start_point, goals):
    l = []
    # t1 <= DFS_Traversal
    # t2 <= UCS_Traversal
    # t3 <= A_star_Traversal
    t1=list()
    t2=list()
    t3=list()
    t1=DFS_Traversal(cost, start_point, goals)
    t2=UCS_Traversal(cost, start_point, goals)
    t3=A_star_Traversal(cost, heuristic, start_point, goals)
    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l