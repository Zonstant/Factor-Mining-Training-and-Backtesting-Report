from gp_set import *

# 函数符集 FFF 内的函数可以为：
# \quad\quad 布尔运算符： AND、OR、NOT 等
# \quad\quad 条件表达式： If-then-else、Switch-case 等
# \quad\quad 循环表达式： Do-until、While-do、For-do 等


name2node={'x':t_x,'abs':f_abs,'sin':f_sin,'cos':f_cos,'sqrt':f_sqrt,'log':f_log,'exp':f_exp,'avg':f_avg,'max':f_max,'min':f_min,'lag':f_lag,'momentum':f_momentum,'rsi':f_rsi,'roc':f_roc,'add':f_add,'sub':f_sub,'mul':f_mul,'div':f_div}

class tree():
    def __init__(self):
        self.root=random.choice(f_list)()
        self.fitness=-1
        self.train_ic=-1
        self.test_ic=-1
        self.test_ir=-1
        self.train_ics=[]
    def full(self,max_depth):
        self.root.full(1,max_depth)
        self.update_depth()
    def grow(self,max_depth):
        self.root.grow(1,max_depth)
        self.update_depth()
    def forward(self,input):
        return self.root.forward(input,1)[0]
    def update_depth(self):
        self.root.update_depth(1)
    def express(self):
        return self.root.express()
    def look(self):
        return self.root.look()
    def save(self):
        return [self.root.save(),self.fitness,self.train_ic,self.test_ic,self.test_ir,self.train_ics]
    def load(self,data):
        self.root=node_load(data[0])
        self.fitness=data[1]
        self.train_ic=data[2]
        self.test_ic=data[3]
        self.test_ir=data[4]
        #self.train_ics=data[5]
    def copy(self):
        s=self.save()
        new_tree=tree()
        new_tree.load(s)
        new_tree.update_depth()
        return new_tree


def obtain_pos(node,parent):
    if isinstance(parent,F_one):
        return 'child'
    if isinstance(parent,F_two):
        if parent.lchild==node:
            return 'lchild'
        else:
            return 'rchild'

def put_pos(node,parent,position):
    node.parent=parent
    if position=='child':
        parent.child=node
    elif position=='lchild':
        parent.lchild=node
    else:
        parent.rchild=node

#！！！如果选择的不是最大的分支，那么不一定要grow到底
def mutation(r,max_depth):
    off=r.copy()
    n_list=off.look()
    node=random.choice(n_list)
    while node.parent is None or (isinstance(node.parent,F_one_n) and node.parent.rchild==node) or node.depth==max_depth:
        node=random.choice(n_list)
    parent=node.parent
    position=obtain_pos(node,parent)
    t=tree()
    t.grow(max_depth-node.depth+1)
    node=t.root
    put_pos(node,parent,position)
    if off.root.get_depth()>max_depth+1:
        return None
    return off

def crossover(r1,r2,max_depth):
    o1=r1.copy()
    o2=r2.copy()
    n_list1=o1.look()
    n_list2=o2.look()
    f1=False
    f2=False
    while (not f1) and (not f2):
        n1=random.choice(n_list1)
        n2=random.choice(n_list2)
        if n1.parent is None or n2.parent is None or (isinstance(n1.parent,F_one_n) and n1.parent.rchild==n1) or (isinstance(n2.parent,F_one_n) and n2.parent.rchild==n2):
            continue
        #n2->n1，保留n1
        if n1.depth+n2.get_depth()<=max_depth+1:
            f1=True
        #n1->n2，保留n2        
        if n2.depth+n1.get_depth()<=max_depth+1:
            f2=True
    p1=n1.parent
    pos1=obtain_pos(n1,p1)
    p2=n2.parent
    pos2=obtain_pos(n2,p2)
    put_pos(n1,p2,pos2)
    put_pos(n2,p1,pos1)
    if not f1:
        o1=None
    if not f2:
        o2=None
    return o1,o2

def k_tournament(pop,k):
    offs = random.sample(pop, k)
    return max(offs, key=lambda x: x.fitness)

def selection(pop,popSize,k):
    repeat=set()
    off=[]
    while len(off)<popSize:
        a=k_tournament(pop,k)
        s=a.express()
        if not repeat.__contains__(s):
            off.append(a)
            repeat.add(s)
    return off