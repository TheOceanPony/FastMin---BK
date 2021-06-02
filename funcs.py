
import numpy as np
from random import choice
import matplotlib.pyplot as plt

from cv2 import imread, resize, cvtColor, COLOR_BGR2GRAY


def import_img(f_name, bw=True, newshape=False):

    # Color convert
    if bw:
        img = cvtColor( imread(f_name), COLOR_BGR2GRAY)
    else:
        img = imread(f_name)

    # Resize
    if newshape != False:
        img = resize(img, newshape)

    print('Input size: ',img.shape)
    print(f"dtype: {img.dtype} | max: {np.max(img)} | min: {np.min(img)}")

    #plt.subplots(figsize=(10, 10))
    plt.imshow(img, cmap='gray')

    return img.astype(np.int32)


def res_check(img):

    img_ = np.copy(img)
    img_ *= 0

    ind = np.where(img > 128)
    img_[ind[0], ind[1]] = 255
        
    plt.imshow(img_, cmap='gray')
    

def show_tree(Tree, shape):
    
    res = np.zeros(Tree.size-2, dtype=np.uint8)
    
    for i in range(0, Tree.size-2):
        
        if Tree[i+1] == 1 or Tree[i+1] == -1:
            res[i] = 0
            
        if Tree[i+1] == 0:
            res[i] = 255
            
    res = np.reshape(res, shape)
    #print(res)
    plt.imshow(res, cmap='gray', vmin=0, vmax=255)



# Graph structure
####################################

def init_g(img, scale):
    
    h, w = img.shape[:2]
    img = img.flatten()

    g = np.zeros((h*w + 2, h*w + 2), dtype = np.int32)

    g[0, 1:-1] = img
    g[1:-1, -1] = 255 - img

    for i in range(h):
        for j in range(w):
            index = j + i*w + 1
            # left
            if j > 0:
                left = j - 1 + i*w + 1
                g[index, left] = scale
            # right
            if j < w - 1:
                right = j + 1 + i*w + 1
                g[index, right] = scale
            # up
            if i > 0:
                up = j + (i-1)*w + 1
                g[index, up] = scale
            # down
            if i < h - 1:
                down = j + (i+1)*w + 1
                g[index, down] = scale
                
    return g


def init_N(g):
    
    N = []
    g_ = g + g.T
    
    for i in range(0, g.shape[0]):
        N.append( np.where( g_[i] > 0 )[0].tolist() )
        
    return N



# Growth stage
####################################

def tree_cap(p,q, G_f, Tree):

    if Tree[p] == 0:
        return G_f[p][q]
    if Tree[p] == 1:
        return G_f[q][p]
    else:
        #print(f"!!!!! Tree[p] : {Tree[p]} ")
        return None #sG_f[p][q]

# ???
def restore_path(p,q, Parent):
    
    S_path = [p]
    v_ = Parent[p]
    while v_ != -1:
        S_path.append(v_)
        v_ = Parent[v_]
    
    T_path = [q]
    v_ = Parent[q]
    while v_ != -1:
        T_path.append(v_)
        v_ = Parent[v_]
    
    return S_path[::-1] + T_path


def Growth_stage(Tree, Parent, A, G_f, N):

    while A != []:

        p = A[0]

        for q in N[p]:
            if tree_cap(p,q, G_f,Tree) > 0:

                if Tree[q] == -1:
                    Tree[q] = Tree[p]
                    Parent[q] = p

                if Tree[q] != -1 and Tree[q] != Tree[p]:
                    path = restore_path(p,q, Parent)[::-1]
                    #print(f"path: {path}")
                    return path, Tree, Parent, A

        A.pop(0)

    print("No path")
    return None, Tree, Parent, A


# Augmentation stage
####################################

def find_path_bottleneck(path, G_f):
    
    v = path[0]
    delta_f = np.inf
    
    for v_next in path[1:]:    
        if delta_f > G_f[v][v_next]:
            delta_f = G_f[v][v_next]
        v = v_next
        
    return delta_f


def update_residual_graph(path, delta_f, G_f):
    
    v = path[0]

    for v_next in path[1:]:
        G_f[v][v_next] -= delta_f
        v = v_next

    return G_f


def Augmentation_stage(path, Tree, Parent, O, G_f):

    delta_f = find_path_bottleneck(path, G_f)
    #print(f"delta_f: {delta_f}")

    G_f = update_residual_graph(path, delta_f, G_f)

    p = path[0]
    for q in path[1:]:
        #print(f">>> {p} -> {q}")
        if G_f[p][q] == 0:
            
            if Tree[p] == 0 and Tree[q] == 0:
                Parent[q] = -1
                O.append(q)
            
            if Tree[p] == 1 and Tree[q] == 1:
                Parent[p] = -1
                O.append(p)
            
        p = q

    return  Parent, O, G_f


# Adoption stage
####################################

def valid_origin(q, Parent):
    
    s,t = 0, Parent.size-1

    v_ = q
    if v_ in [s,t]:
        return True
    else:
        v_ = Parent[q]
        while v_ not in [q, s, t, -1]:
            v_ = Parent[v_]
        
    if v_ in [s,t]:
        return True
    else:
        return False


def valid_parent(q, p, Tree, Parent, G_f):
    # checks if q is a valid parent for p
    if Tree[q] != Tree[p]:
        return False

    if not valid_origin(q, Parent):
        return False

    if tree_cap(q, p, G_f, Tree) <= 0:
        return False

    return True


def Adoption_stage(Tree, Parent, O, A, N, G_f):

    while O != []:

        o = O[0]
        O.pop(0)

        for q in N[o]:

            if valid_parent(q, o, Tree, Parent, G_f):
                Parent[o] = q
                return Tree, Parent, O, A

        # parent not found
        for q in N[o]:
            if Tree[q] == Tree[o] and Tree[q] != -1:

                if tree_cap(q,o, G_f, Tree) > 0:
                    A.append(q)
                
                if Parent[q] == o:
                    O.append(q)
                    Parent[q] = -1

        Tree[o] = -1
        if o in A:
            A.remove(o)

    return Tree, Parent, O, A


                    







