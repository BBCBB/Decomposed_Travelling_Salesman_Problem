#%%
import random
import math
from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import statistics
import sys
from scipy.spatial import ConvexHull, convex_hull_plot_2d


# INPUT -----------------------------------------------------------------------
Point = namedtuple("Point", ['x', 'y'])      # function for calculating the length between two points
def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

f = open(r"C:datafile2.txt","r")

input_data = ''.join(f.readlines())
lines = input_data.split('\n')

nc = int(lines[0])

points = []
for i in range(1, nc+1):
    line = lines[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))
 
# Auxiliary Functions ---------------------------------------------------------

# AF 1
def unique(list1):
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    a=[]
    for x in unique_list:
        a.append(x)
    return a


# DECOMPOSITION ---------------------------------------------------------------
x=[]
y=[]
for i in range(0,nc):
    x.append(points[i][0])
    y.append(points[i][1])
    xy=np.array([x,y])
xm=statistics.median(x)
ym=statistics.median(y)


P1=[]
P2=[]
P3=[]
P4=[]

for i in range(len(points)):
    if points[i][0]<=xm and points[i][1]<=ym:
        P1.append(points[i])
    elif points[i][0]<=xm and points[i][1]>ym:
        P2.append(points[i])
    elif points[i][0]>xm and points[i][1]>ym:
        P3.append(points[i])
    else:
        P4.append(points[i])  
ps=range(len(P1+P2+P3+P4))

G=nx.DiGraph()
G.add_nodes_from(ps)

pos = {}
for i in range(len(P1)+len(P2)+len(P3)+len(P4)):
    pos[i]=[]
for i in range(len(P1)):
    pos[i].append(P1[i][0])
    pos[i].append(P1[i][1])

for i in range(len(P1),len(P1)+len(P2)):
    pos[i].append(P2[i-len(P1)][0])
    pos[i].append(P2[i-len(P1)][1])

for i in range(len(P1)+len(P2),len(P1)+len(P2)+len(P3)):
    pos[i].append(P3[i-len(P1)-len(P2)][0])
    pos[i].append(P3[i-len(P1)-len(P2)][1])

for i in range(len(P1)+len(P2)+len(P3),len(P1)+len(P2)+len(P3)+len(P4)):
    pos[i].append(P4[i-len(P1)-len(P2)-len(P3)][0])
    pos[i].append(P4[i-len(P1)-len(P2)-len(P3)][1])
    
fig1, ax1 = plt.subplots(figsize=(10, 10))

nx.draw_networkx_nodes(G,pos,node_size=10,node_color='purple',ax=ax1)
ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

deltatmp=float("inf")
for t in range(1,400):
    m=0.0025*t
    Pp1=[]
    Pp2=[]
    Pp3=[]
    Pp4=[]
    for i in range(len(points)):
        
        b1=points[i][1]-m*points[i][0]-ym+m*xm
        b2=points[i][1]+1/m*points[i][0]-ym-1/m*xm
        if b1<0 and b2>=0:
            Pp1.append(points[i])
        elif b1>=0 and b2>0:
            Pp2.append(points[i])
        elif b1>=0 and b2<0:
            Pp3.append(points[i])
        else:
            Pp4.append(points[i])
    delta=abs(len(Pp1)-nc/4)+abs(len(Pp2)-nc/4)+abs(len(Pp3)-nc/4)+abs(len(Pp4)-nc/4)
    
    if delta<deltatmp:
        tmp=[Pp1,Pp2,Pp3,Pp4]
        mm=m
        dev=delta
        deltatmp=delta
Pp1=tmp[0]
Pp2=tmp[1]
Pp3=tmp[2]
Pp4=tmp[3]     


print('set1: {} - set2: {} - set3: {} - set4: {} - deviation %: {}'.format(len(Pp1),len(Pp2),len(Pp3),len(Pp4),round(dev/nc,3)*100))

# CONVEX HULL and Links Between Subproblems
p1=np.asarray(Pp1)
p2=np.asarray(Pp2)
p3=np.asarray(Pp3)
p4=np.asarray(Pp4)
p11=[]

xy1=[]
hull = ConvexHull(p1)
for simplex in hull.simplices:
    plt.plot(p1[simplex, 0], p1[simplex, 1], 'g-')
    plt.plot(p1[simplex,0], p1[simplex,1], 'o',markersize=10)
    p11.append([p1[simplex,0],p1[simplex,1]])
for i in range(len(p11)): 
    xy1.append(Point(p11[i][0][0],p11[i][1][0]))
    xy1.append(Point(p11[i][0][1],p11[i][1][1]))

cvx1=[]
for i in unique(xy1):
    cvx1.append(points.index(i))
    
p22=[]
xy2=[]
hull = ConvexHull(p2)
for simplex in hull.simplices:
    plt.plot(p2[simplex, 0], p2[simplex, 1], 'b-')
    plt.plot(p2[simplex,0], p2[simplex,1], 'o',markersize=10)
    p22.append([p2[simplex,0],p2[simplex,1]])
for i in range(len(p22)): 

    xy2.append(Point(p22[i][0][0],p22[i][1][0]))
    xy2.append(Point(p22[i][0][1],p22[i][1][1]))

cvx2=[]
for i in unique(xy2):
    cvx2.append(points.index(i))

p33=[]
xy3=[]
hull = ConvexHull(p3)
for simplex in hull.simplices:
    plt.plot(p3[simplex, 0], p3[simplex, 1], 'y-')
    plt.plot(p3[simplex,0], p3[simplex,1], 'o',markersize=10)
    p33.append([p3[simplex,0],p3[simplex,1]])
    
for i in range(len(p33)): 

    xy3.append(Point(p33[i][0][0],p33[i][1][0]))
    xy3.append(Point(p33[i][0][1],p33[i][1][1]))
cvx3=[]
for i in unique(xy3):
    cvx3.append(points.index(i))

hull = ConvexHull(p4)
p44=[]
x4=[]
y4=[]
xy4=[]
for simplex in hull.simplices:
    plt.plot(p4[simplex, 0], p4[simplex, 1], 'k-')
    plt.plot(p4[simplex,0], p4[simplex,1], 'o',markersize=10)
    p44.append([p4[simplex,0],p4[simplex,1]])
    
for i in range(len(p44)): 
    xy4.append(Point(p44[i][0][0],p44[i][1][0]))
    xy4.append(Point(p44[i][0][1],p44[i][1][1]))

cvx4=[]
for i in unique(xy4):
    cvx4.append(points.index(i))  

M=float("inf")
onetwo=[0,0]
linklength=0
sval=float("inf")
for i,j in enumerate(cvx1):
    for k,l in enumerate(cvx2):
        obj=length(points[j],points[l])
        if obj<M:
            M=obj
            onetwo[0]=j
            onetwo[1]=l
linklength=linklength+M
            
plt.plot(np.array([points[onetwo[0]][0],points[onetwo[1]][0]]), np.array([points[onetwo[0]][1],points[onetwo[1]][1]]), 'g-',linewidth=5)         
M=float("inf")
twothree=[0,0]
for i,j in enumerate(cvx2):
    for k,l in enumerate(cvx3):
        obj=length(points[j],points[l])
        if obj<M and j!=onetwo[1]:
            M=obj
            twothree[0]=j
            twothree[1]=l
linklength=linklength+M
plt.plot(np.array([points[twothree[0]][0],points[twothree[1]][0]]), np.array([points[twothree[0]][1],points[twothree[1]][1]]), 'g-',linewidth=5)           
M=float("inf")
threefour=[0,0]
for i,j in enumerate(cvx3):
    for k,l in enumerate(cvx4):
        obj=length(points[j],points[l])
        if obj<M and j!=twothree[1]:
            M=obj
            threefour[0]=j
            threefour[1]=l
linklength=linklength+M
plt.plot(np.array([points[threefour[0]][0],points[threefour[1]][0]]), np.array([points[threefour[0]][1],points[threefour[1]][1]]), 'g-',linewidth=5)
M=float("inf")
fourone=[0,0]
for i,j in enumerate(cvx4):
    for k,l in enumerate(cvx1):
        obj=length(points[j],points[l])
        if obj<M and j!=threefour[1] and l!=onetwo[0]:
            M=obj
            fourone[0]=j
            fourone[1]=l
linklength=linklength+M
plt.plot(np.array([points[fourone[0]][0],points[fourone[1]][0]]), np.array([points[fourone[0]][1],points[fourone[1]][1]]), 'g-',linewidth=5)
sub1=[0,0]
sub1[0]=Pp1.index(points[fourone[1]])
sub1[1]=Pp1.index(points[onetwo[0]])

sub2=[0,0]
sub2[0]=Pp2.index(points[onetwo[1]])
sub2[1]=Pp2.index(points[twothree[0]])

sub3=[0,0]
sub3[0]=Pp3.index(points[twothree[1]])
sub3[1]=Pp3.index(points[threefour[0]])

sub4=[0,0]
sub4[0]=Pp4.index(points[threefour[1]])
sub4[1]=Pp4.index(points[fourone[0]])


#%% SIMULATED ANNEALING
def simulan(cities,nodeCount,sub):
    # AF 2 -----------------------------------
    def obj(s):                               # OBJECTIVE VALUE
        objective=0
        for j in range(nodeCount-1):
            l= length(cities[s[j]],cities[s[j+1]])
            objective = objective + l
        return objective

    # AF 3 -----------------------------------
    def deltap(y,w,z,s):#a function to calculate the IMPROVEMENT or DETERIORATION..
        imp=0           # ... in the Objective value based on each swap
        if w==0:
            imp=length(cities[s[y-1]],cities[s[y]])+length(cities[s[z]],cities[s[z+1]])  
        elif w==-1:
            imp=length(cities[s[y-1]],cities[s[y]])+length(cities[s[z-1]],cities[s[z]])  
        else:
            imp=length(cities[s[y-1]],cities[s[y]])+length(cities[s[z-1]],cities[s[z]])+\
                length(cities[s[w-1]],cities[s[w]])
        return imp
    Dist_mat=[[0 for i in range(nodeCount)] for i in range(nodeCount)]       #calculating Distance Matrix to reduce the search space when choosing an arc to swap randomly
    for i in range(nodeCount):
        for j in range(i,nodeCount):
           Dist_mat[i][j]=abs(cities[i].x-cities[j].x)+abs(cities[i].y-cities[j].y)
           Dist_mat[j][i]=Dist_mat[i][j]
    
    solution=[sub[0]]                                                             #Generating an initial solution   
    for i in solution:
        l=float("Inf") 
        for j in range(0,nodeCount):
            if j in sub or j in solution:
                continue
            else:
                s=abs(cities[i].x-cities[j].x)+abs(cities[i].y-cities[j].y)
                l=min(l,s)
                if l==s:
                    ngbr=j
        if ngbr in solution:
            continue
        else:
            solution.append(ngbr)
        if len(solution)==nodeCount:
            break

    solution.append(sub[1])                                               # transforming the initial solution to a cycle
    # determining different parameters based the size of the problem
    if nodeCount<50:
        t=5                                                         # initial temperature of the system
        maxiter=500                                                  # number of different temperatures the system will experience
        m=400
        trenew=1000
        cp=0.99
        ngbr=0.3
    elif nodeCount<150:
        t=10                                                        # initial temperature of the system
        maxiter=600                                                  # number of different temperatures the system will experience
        m=650
        trenew=300
        cp=0.99
        ngbr=0.3
    elif nodeCount<500:
        t=30                                                         # initial temperature of the system
        maxiter=600                                                  # number of different temperatures the system will experience
        m=700
        trenew=500
        cp=0.95
        ngbr=0.15
    elif nodeCount<1000:
        t=20                                                        # initial temperature of the system
        maxiter=800                                                  # number of different temperatures the system will experience
        m=900
        trenew=2000
        cp=0.99
        ngbr=0.08
    elif nodeCount<2000:
        t=20                                                        # initial temperature of the system
        maxiter=900                                                  # number of different temperatures the system will experience
        m=1000
        trenew=3000
        cp=0.99
        ngbr=0.06
    else:
        t=20                                                        # initial temperature of the system
        maxiter=800                                                  # number of different temperatures the system will experience
        m=900
        trenew=4000
        cp=0.99
        ngbr=0.04
    t0=t
    temprec=[]
    start_time = time.time()                        # Establishing the start time to calculate the run time
    objval=obj(solution)                            # calcuclating the objective function based on the initial solution
    bestval=objval                                  # determining an initial value for the best objective value achieved so far
    bestsol=solution.copy()                         # determining an initial value for the best solution achieved so far
    objrec=[]                                       # storing a record of all objective values obtained through the algorithm
    counter=0
    for iteration in range(1,maxiter+1):
        for j in range(m):
            objrec.append(objval)
        #------------------------------   2 OPT   -------------------------------------   
            ind1=random.randint(1,nodeCount-1)
            num=solution[ind1-1]
            order_sel=Dist_mat[num]
            order=sorted(range(len(order_sel)), key=lambda k: order_sel[k])
            
            ind2=order[random.randint(1,int(len(order)*ngbr))]
            ind2=solution.index(ind2)
            if ind2==0:
                ind2=1
                
            inds=sorted([ind1,ind2])
            if inds[1]==nodeCount-1 and inds[0]==nodeCount-2:
                inds[0]=nodeCount-3
            
            rp=random.uniform(0,1)
            if inds[1]-inds[0]==1:
                oo=deltap(inds[0],0,inds[1],solution)
                x=solution.copy()
                a=x.pop(inds[0])
                x.insert(inds[1],a)
                ox=deltap(inds[0],0,inds[1],x)
                if ox>oo:
                    pr=math.exp((int(oo-ox)-1)/t)
                else:
                    pr=0
                if ox<=oo or rp<pr:
                    a=solution.pop(inds[0])
                    solution.insert(inds[1],a)
                    objval=objval-oo+ox
                                
            else:
                oo=deltap(inds[0],-1,inds[1],solution)
                x=solution.copy()
                x[inds[0]:inds[1]]=x[inds[1]-1:inds[0]-1:-1]
                ox=deltap(inds[0],-1,inds[1],x)
                if ox>oo:
                    pr=math.exp((int(oo-ox)-1)/t)
                else:
                    pr=0
                if ox<=oo or rp<pr:
                    solution[inds[0]:inds[1]]=solution[inds[1]-1:inds[0]-1:-1]
                    objval=objval-oo+ox
        #-------------------------------   3 OPT   ------------------------------------
        
            ind1=random.randint(1,nodeCount-1)
            num=solution[ind1-1]
            order_sel=Dist_mat[num]
            order=sorted(range(len(order_sel)), key=lambda k: order_sel[k])
            order.remove(solution[ind1])
            flag=0
            while flag==0:
                tbc3opt=random.sample(order[1:int(len(order)*ngbr)],2)
                ind2=solution.index(tbc3opt[0])
                ind3=solution.index(tbc3opt[1])
                
                if ind2!=0 and ind3 !=0:
                    flag=1
            inds=sorted([ind1,ind2,ind3])
            oo=deltap(inds[0],inds[1],inds[2],solution)
            rp=random.uniform(0,1)
            if inds[2]-inds[0]==2:
                b=solution.copy()
                a=b.pop(inds[0])
                b.insert(inds[1],a)
                ob=deltap(inds[0],inds[1],inds[2],b)
                if ob>oo:
                    pr=math.exp((int(oo-ob)-1)/t)
                else:
                    pr=0
                if ob<=oo or rp<=pr:
                    a=solution.pop(inds[0])
                    solution.insert(inds[1],a)
                    objval=objval-oo+ob
                
                
            elif inds[2]-inds[1]!=1 and inds[1]-inds[0]==1:
                b=solution.copy()
                c=solution.copy()
                a=b.pop(inds[0])
                b.insert(inds[2]-1,a)
                ob=deltap(inds[0],inds[2]-1,inds[2],b)
                c[inds[0]:inds[2]]=c[inds[2]-1:inds[0]-1:-1]
                oc=deltap(inds[0],inds[2]-1,inds[2],c)
                op=[ob,oc]
                if min(op)>oo:
                    pr=math.exp((int(oo-min(op))-1)/t)
                else:
                    pr=0
                if min(op)<=oo or rp<pr:
                    if ob>oc:
                        solution[inds[0]:inds[2]]=solution[inds[2]-1:inds[0]-1:-1]
                        objval=objval-oo+oc
                    else:
                        a=solution.pop(inds[0])
                        solution.insert(inds[2]-1,a)
                        objval=objval-oo+ob
                   
                   
            elif inds[2]-inds[1]==1 and inds[1]-inds[0]!=1:
                b=solution.copy()
                a=b.pop(inds[1])
                b.insert(inds[0],a)
                ob=deltap(inds[0],inds[0]+1,inds[2],b)
                c=solution.copy()
                c[inds[0]:inds[1]]=c[inds[1]-1:inds[0]-1:-1]
                oc=deltap(inds[0],inds[1],inds[2],c)
                d=solution.copy()
                d[inds[0]:inds[2]]=d[inds[2]-1:inds[0]-1:-1]
                od=deltap(inds[0],inds[0]+1,inds[2],d)
                op=[ob,oc,od]
                if min(op)>oo:
                    pr=math.exp((int(oo-min(op))-1)/t)
                else:
                    pr=0
                if min(op)<=oo or rp<pr:
                    if min(op) == od:
                        solution[inds[0]:inds[2]]=solution[inds[2]-1:inds[0]-1:-1] 
                        objval=objval-oo+od
                    elif min(op) == oc:
                        solution[inds[0]:inds[1]]=solution[inds[1]-1:inds[0]-1:-1]
                        objval=objval-oo+oc
                    else:
                        a=solution.pop(inds[1])
                        solution.insert(inds[0],a)
                        objval=objval-oo+ob
            else:
                b=solution.copy()
                b[inds[0]:inds[2]]=b[inds[2]-1:inds[0]-1:-1]
                ob=deltap(inds[0],inds[0]+inds[2]-inds[1],inds[2],b)
                
                c=solution.copy()
                c[inds[1]:inds[2]]=c[inds[2]-1:inds[1]-1:-1]
                a=c[inds[0]:inds[1]].copy()
                del c[inds[0]:inds[1]]
                c[inds[0]+inds[2]-inds[1]:inds[0]+inds[2]-inds[1]]=a
                oc=deltap(inds[0],inds[0]+inds[2]-inds[1],inds[2],c)
                
            
                d=solution.copy()    
                d[inds[0]:inds[1]]=d[inds[1]-1:inds[0]-1:-1]
                d[inds[1]:inds[2]]=d[inds[2]-1:inds[1]-1:-1]
                od=deltap(inds[0],inds[1],inds[2],d)
                
                
                e=solution.copy()               #2-OPT
                e[inds[0]:inds[1]]=e[inds[1]-1:inds[0]-1:-1]
                oe=deltap(inds[0],inds[1],inds[2],e)
                
            
                f=solution.copy()               #2-OPT
                f[inds[1]:inds[2]]=f[inds[2]-1:inds[1]-1:-1]
                of=deltap(inds[0],inds[1],inds[2],f)
                
           
                g=solution.copy()
                a=g[inds[0]:inds[1]].copy()
                del g[inds[0]:inds[1]]
                g[inds[0]+inds[2]-inds[1]:inds[0]+inds[2]-inds[1]]=a
                og=deltap(inds[0],inds[0]+inds[2]-inds[1],inds[2],g)
                
            
                h=solution.copy()
                h[inds[0]:inds[1]]=h[inds[1]-1:inds[0]-1:-1]
                a=h[inds[0]:inds[1]].copy()
                del h[inds[0]:inds[1]]
                h[inds[0]+inds[2]-inds[1]:inds[0]+inds[2]-inds[1]]=a
                oh=deltap(inds[0],inds[0]+inds[2]-inds[1],inds[2],h)
                
                op=[ob,oc,od,oe,of,og,oh]
                if min(op)>oo:
                    pr=math.exp((int(oo-min(op))-1)/t)
                else:
                    pr=0
                if min(op)<=oo or rp<pr:
                    if min(op)==ob:
                        solution[inds[0]:inds[2]]=solution[inds[2]-1:inds[0]-1:-1]
                        objval=objval-oo+ob
                    elif min(op)==oc:
                        solution[inds[1]:inds[2]]=solution[inds[2]-1:inds[1]-1:-1]
                        a=solution[inds[0]:inds[1]].copy()
                        del solution[inds[0]:inds[1]]
                        solution[inds[0]+inds[2]-inds[1]:inds[0]+inds[2]-inds[1]]=a
                        objval=objval-oo+oc
                    elif min(op)==od:
                        solution[inds[0]:inds[1]]=solution[inds[1]-1:inds[0]-1:-1]
                        solution[inds[1]:inds[2]]=solution[inds[2]-1:inds[1]-1:-1]
                        objval=objval-oo+od
                    elif min(op)==oe:
                        solution[inds[0]:inds[1]]=solution[inds[1]-1:inds[0]-1:-1]
                        objval=objval-oo+oe
                    elif min(op)==of:
                        solution[inds[1]:inds[2]]=solution[inds[2]-1:inds[1]-1:-1]
                        objval=objval-oo+of
                    elif min(op)==og:
                        a=solution[inds[0]:inds[1]].copy()
                        del solution[inds[0]:inds[1]]
                        solution[inds[0]+inds[2]-inds[1]:inds[0]+inds[2]-inds[1]]=a
                        objval=objval-oo+og
                    elif min(op)==oh:
                        solution[inds[0]:inds[1]]=solution[inds[1]-1:inds[0]-1:-1]
                        a=solution[inds[0]:inds[1]].copy()
                        del solution[inds[0]:inds[1]]
                        solution[inds[0]+inds[2]-inds[1]:inds[0]+inds[2]-inds[1]]=a
                        objval=objval-oo+oh
        
            counter+=1
            if objval<bestval:
                bestsol=solution.copy()
                bestval=objval
            temprec.append(t)
        if counter> trenew and max(objrec[counter-trenew:counter])-min(objrec[counter-trenew:counter])==0:
            t=min(t0,t*1.5)
            
        else:
            t=cp*t
        
    
    sol=bestsol.copy()
    
    output_data = '%.2f' % bestval + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, sol))        
    print(output_data)
    print(' Execution time is {} seconds'.format(round(time.time() - start_time,5)))
    return solution,iteration,bestval

solution,iteration,bestval=simulan(Pp1,len(Pp1),sub1)    # Subproblem 1
solution2,iteration,bestval2=simulan(Pp2,len(Pp2),sub2)  # Subproblem 2
solution3,iteration,bestval3=simulan(Pp3,len(Pp3),sub3)  # Subproblem 3
solution4,iteration,bestval4=simulan(Pp4,len(Pp4),sub4)  # Subproblem 4

#%% PLOT
nx.draw_networkx_edges(G,pos, width=1,edge_color='black',ax=ax1)
ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
nodes=range(len(solution))
G.add_nodes_from(nodes)
edges=[]
for i in range(len(Pp1)-1):  
    edges.append((solution[i],solution[i+1]))
G.add_edges_from(edges)
pos = {}
for i in range(len(Pp1)):
    pos[i]=[]
for i in range(len(Pp1)):
    pos[i].append(Pp1[i][0])
    pos[i].append(Pp1[i][1])


for i,j in enumerate(solution2):
    solution2[i]=j+len(Pp1)
for i,j in enumerate(solution3):
    solution3[i]=j+len(Pp1)+len(Pp2)
for i,j in enumerate(solution4):
    solution4[i]=j+len(Pp1)+len(Pp2)+len(Pp3)
nodes=range(len(points))
G.add_nodes_from(nodes)
edges=[]
for i in range(len(Pp1)-1):
    edges.append((solution[i],solution[i+1]))
for i in range(len(Pp2)-1):
    edges.append((solution2[i],solution2[i+1]))
for i in range(len(Pp3)-1):
    edges.append((solution3[i],solution3[i+1]))
for i in range(len(Pp4)-1):
    edges.append((solution4[i],solution4[i+1]))
G.add_edges_from(edges)
pos = {}
for i in range(len(Pp1)):
    pos[i]=[]
for i in range(len(Pp1)):
    pos[i].append(Pp1[i][0])
    pos[i].append(Pp1[i][1])
    
for i in range(len(Pp1),len(Pp1)+len(Pp2)):
    pos[i]=[]
for i in range(len(Pp1),len(Pp1)+len(Pp2)):
    pos[i].append(Pp2[i-len(Pp1)][0])
    pos[i].append(Pp2[i-len(Pp1)][1])
    
for i in range(len(Pp1)+len(Pp2),len(Pp1)+len(Pp2)+len(Pp3)):
    pos[i]=[]
for i in range(len(Pp1)+len(Pp2),len(Pp1)+len(Pp2)+len(Pp3)):
    pos[i].append(Pp3[i-len(Pp1)-len(Pp2)][0])
    pos[i].append(Pp3[i-len(Pp1)-len(Pp2)][1])
    
for i in range(len(Pp1)+len(Pp2)+len(Pp3),len(Pp1)+len(Pp2)+len(Pp3)+len(Pp4)):
    pos[i]=[]
for i in range(len(Pp1)+len(Pp2)+len(Pp3),len(Pp1)+len(Pp2)+len(Pp3)+len(Pp4)):
    pos[i].append(Pp4[i-len(Pp1)-len(Pp2)-len(Pp3)][0])
    pos[i].append(Pp4[i-len(Pp1)-len(Pp2)-len(Pp3)][1])
    
nx.draw_networkx_edges(G,pos, width=1,edge_color='black',ax=ax1)
ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
print(' Objective Value is {} '.format(bestval+bestval2+bestval3+bestval4+linklength))