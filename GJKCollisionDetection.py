import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import collections

from sklearn import preprocessing

def VectorNormalized(A):
    Ax,Ay,Az = A
    if np.sqrt(Ax**2 + Ay**2 + Az**2) == 0:
        print('vector of length[0]?!')
        return np.array([0,0,0]).flatten()
    else:
        VectN = np.sqrt(Ax**2 + Ay**2 + Az**2)
        vn = np.array([Ax,Ay,Az])/VectN
        return vn

### Support functions:

def tripleProd(A,B,C):

    X = np.cross(A,B)

    return np.cross(X,A)


def lineCase(simplex,d):

    B,A = simplex
    AB,AO = B-A,ORIGIN - A
    ABperp = tripleProd(AB,AO,AB)

    d = ABperp

    result = False

    return result,d

def triangleCase(simplex,d):

    ORIGIN = np.array([0.0,0.0,0.0])

    C,B,A = simplex

    AB,AC,AO = B-A,C-A,ORIGIN - A

    ABperp = tripleProd(AC,AB,AB)
    ACperp = tripleProd(AB,AC,AC)

    if np.dot(ABperp,AO) > 0 :
        simplex[-3] = B
        simplex[-2] = A
        print(f'ABperpendicular:\n{ABperp}')
        d = ABperp.flatten()

        result = False

        return result,d # origin in area Aab find new a
    if np.dot(ACperp,AO) > 0 :
        simplex[-3] = C
        simplex[-2] = A
        #d = ACperp.flatten()
        print(f'ACperpendicular:\n{ACperp}')
        result = False

        return result,d # origin in area Aac find new a

    result = True

    return result,d # origin is in area Acba

def handleSimplex(simplex,d):

    if len(simplex) == 2:
        result,d = lineCase(simplex,d)
        return result,d
    result,d = triangleCase(simplex,d)
    return result,d

def nSimplexCentroid(simplex):

    simplex = np.array(simplex)

    n = len(simplex[0,:])

    C = np.zeros(n)

    for i in range(n):

        C[i] = np.sum(simplex[:,i])

    return C * 1/(n+1)

def support(Shape1,Shape2,d):
     # a support vector ?

    #Shape1 = np.asarray(Shape1)
    #Shape2 = np.asarray(Shape2)

    d = d.flatten()
    P1 = [np.dot(Shape1[n],d) for n in range(len(Shape1))]
    P2 = [np.dot(Shape2[n],-d) for n in range(len(Shape2))]
    i = np.where(P1 == np.amax(P1))
    j = np.where(P2 == np.amax(P2))
    v = np.asarray(Shape1[i] - Shape2[j])
    return v

## GJK algorithm

def GJK(Shape1,Shape2):

    ORIGIN = np.array([0.0,0.0,0.0]).flatten()
    i,j = np.shape(Shape1)
    simplex = np.zeros(3,dtype=(float,j))

    print('init')

    # for i in range(len(Shape1)):
    #     Shape1[i] = VectorNormalized(Shape1[i])
    # for j in range(len(Shape2)):
    #     Shape2[j] = VectorNormalized(Shape2[j])


    C = np.array([nSimplexCentroid(Shape1),nSimplexCentroid(Shape2)])
    d = VectorNormalized(C[0] - C[1]) # direction toward other shape
    #print(d) # should be one?

    simplex[-1] = support(Shape1,Shape2,d) # first point?

    #print(f'first point, {simplex[-1]}')

    d = ORIGIN - simplex[-1] # direction towards Origin

    #print(d)

    while True:

        print(f'New direction:\n{d}')

        A = support(Shape1,Shape2,d) # new point in direction of origin

        print(f'New A is:\n{A}')

        if np.dot(A,d) < 0: # does this point pass the origin

            #print('A is not beyond origin')

            T=False

            return False

        # shift points and add new point

        simplex[-3]  = simplex[-2]
        simplex[-2] = simplex[-1]
        simplex[-1] = A

        [testshape,d] = handleSimplex(simplex,d)
        if testshape:
            print('Shapes Collide')

            T=True

            return True
    return T


np.random.seed(12432543)

n = 3 # number of vertices


A = [np.random.uniform(-10,10) for n in range(n)]
B = [np.random.uniform(-10,10) for n in range(n)]
C = [np.random.uniform(-10,10) for n in range(n)]

simplex1 = np.array([[a,b,c] for a,b,c in zip(A,B,C)])

m = 3

A = [np.random.uniform(-10,10) for n in range(m)]
B = [np.random.uniform(-10,10) for n in range(m)]
C = [np.random.uniform(-10,10) for n in range(m)]
simplex2 = np.array([[a,b,c] for a,b,c in zip(A,B,C)])

simplex1=np.array([[0,7,0],[1,0,4],[2,2,0],[-2,3,-5],[2,-3,5]])

simplex2=np.array([[5,0,0],[2,3,4],[2,1,0],[-1,2,-4],[-2,-2,6]])


GJK(simplex1,simplex2)

FigNum = 'Figure'

fig = plt.figure(num=FigNum,figsize = [4,4])
ax = fig.add_axes([0, 0, 1, 1],projection='3d')

ax.set_xlim(-5, 7), ax.set_xticks([])
ax.set_ylim(-5, 7), ax.set_yticks([])
ax.set_zlim(-5, 7), ax.set_zticks([])


#ax.plot3D(simplex1[:,0],simplex1[:,1],simplex2[:,2])



def MakeFrame(simplex):
    FrameX,FrameY,FrameZ = np.zeros((len(simplex)**2,2)),np.zeros((len(simplex)**2,2)),np.zeros((len(simplex)**2,2))
    c=0
    for n in range(0,len(simplex)):
        for m in range(0,len(simplex)):
            FrameX[c,:] = np.asarray([simplex[n,0],simplex[-m,0]])
            FrameY[c,:] = np.asarray([simplex[n,1],simplex[-m,1]])
            FrameZ[c,:] = np.asarray([simplex[n,2],simplex[-m,2]])

            c+=1
    np.vsplit(FrameX,[0,-1])
    np.vsplit(FrameY,[0,-1])
    np.vsplit(FrameZ,[0,-1])

    return FrameX,FrameY,FrameZ

FrameX1,FrameY1,FrameZ1 = MakeFrame(simplex1)

ax.plot_wireframe(FrameX1,FrameY1,FrameZ1,color='tab:green')



FrameX2,FrameY2,FrameZ2 = MakeFrame(simplex2)

ax.plot_wireframe(FrameX2,FrameY2,FrameZ2,color='tab:orange')

def FindFaces(simplex,k):

    print(np.shape(simplex))

    new_simplex = np.zeros((3,3))

    dir = VectorNormalized(nSimplexCentroid(simplex) - simplex[k]) # unit vector pointing toward center.
    P = [np.dot(simplex[m],dir) for m in range(len(simplex))]
    i = np.where(P == np.amax(P))

    AB = simplex[k] - simplex[i]
    AC = nSimplexCentroid(simplex) - simplex[k]
    d = np.cross(AC,AB)# kierunek płaszczyzny ze środkiem figury
    dir = d.flatten()
    P = [np.dot(simplex[m],dir) for m in range(len(simplex))]
    j = np.where(P == np.amax(P))

    new_simplex[0] = simplex[0]
    new_simplex[1] = simplex[i]
    new_simplex[2] = simplex[j]

    ax.plot_trisurf(new_simplex[:,0],new_simplex[:,1],new_simplex[:,2],color='tab:blue')

    # następnie bierzemy krawędź np P0-P1 i powtarzamy

    AB = new_simplex[0] - new_simplex[1]
    AC = new_simplex[0] - new_simplex[2]
    d = VectorNormalized(np.cross(AC,AB))
    dir = d.flatten()
    P = [np.dot(simplex[m],dir) for m in range(len(simplex))]
    k = np.where(P == np.amax(P))

    new_simplex[0] = new_simplex[1]
    new_simplex[1] = new_simplex[2]
    new_simplex[2] = simplex[k]

    ax.plot_trisurf(new_simplex[:,0],new_simplex[:,1],new_simplex[:,2],color='tab:blue')

    AB = new_simplex[0] - new_simplex[1]
    AC = new_simplex[0] - new_simplex[2]
    d = np.cross(AC,AB)
    dir = d.flatten()

    P = [np.dot(simplex[m],dir) for m in range(len(simplex))]
    l = np.where(P == np.amax(P))

    if len(l)>1:
        d = np.cross(AB,AC)
        dir = d.flatten()
        P = [np.dot(simplex[m],dir) for m in range(len(simplex))]
        l = np.where(P == np.amax(P))

    print(f'l:{l}')

    new_simplex[0] = new_simplex[1]
    new_simplex[1] = new_simplex[2]
    new_simplex[2] = simplex[l,:]

    ax.plot_trisurf(new_simplex[:,0],new_simplex[:,1],new_simplex[:,2],color='tab:blue')

    AB = new_simplex[0] - new_simplex[1]
    AC = new_simplex[0] - new_simplex[2]
    d = np.cross(AC,AB)
    dir = d.flatten()
    P = [np.dot(simplex[m],dir) for m in range(len(simplex))]
    m = np.where(P == np.amax(P))

    new_simplex[0] = new_simplex[1]
    new_simplex[1] = new_simplex[2]
    new_simplex[2] = simplex[m,:]

    #print([i],[j],[k],[l],[m])

    ax.plot_trisurf(new_simplex[:,0],new_simplex[:,1],new_simplex[:,2],color='tab:blue')


    return new_simplex

FindFaces(simplex1,0)

#ax.scatter3D(0,0,0)


plt.show()