import math
import numpy as np
import pandas as pd
import random

t_list=None
f1_list=None
f1n_list=None
f2_list=None
f_list=None

t_nodes=['price', 'close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio','pb', 'ps', 'ps_ttm', 'total_share', 'float_share', 'free_share','total_mv', 'circ_mv']

class T_node:
    def __init__(self,name):
        self.name=name
        self.parent=None
        self.depth=-1
    
    def look(self):
        return [self]
    
    def express(self):
        return self.name
    
    def get_depth(self):
        return 1
    
    def update_depth(self,depth):
        self.depth=depth

    def save(self):
        return [self.name,None,None]

class F_node:
    def __init__(self,name):
        self.name=name
        self.parent=None
        self.depth=-1

class F_one(F_node):
    def __init__(self,name):
        super().__init__(name)
        self.child=None
    
    def full(self,depth,maxdepth):
        if depth==maxdepth-1:
            self.child=random.choice(t_list)()
        else:
            self.child=random.choice(f_list)()
            self.child.full(depth+1,maxdepth)
        self.child.parent=self
    
    def grow(self,depth,maxdepth):
        if depth==maxdepth-1:
            self.child=random.choice(t_list)()
        else:
            self.child=random.choice(f_list)()
            self.child.full(depth+1,maxdepth)
        self.child.parent=self
    
    def express(self):
        return self.name+'('+self.child.express()+')'
    
    def update_depth(self,depth):
        #从上至下
        self.depth=depth
        self.child.update_depth(depth+1)
    
    def get_depth(self):
        #从下至上
        return self.child.get_depth()+1
    
    def look(self):
        #!!!
        list=self.child.look()
        list.append(self)
        return list

    def save(self):
        r1=self.child.save()
        return [self.name,r1,None]

class F_two(F_node):
    def __init__(self,name):
        super().__init__(name)
        self.lchild=None
        self.rchild=None
    
    
    def express(self):
        return '('+self.lchild.express()+' '+self.name+' '+self.rchild.express()+')'
    
    def update_depth(self,depth):
        #从上至下
        self.depth=depth
        a=self.lchild.update_depth(depth+1)
        b=self.rchild.update_depth(depth+1)
    
    def get_depth(self):
        #从下至上
        a=self.lchild.get_depth()+1
        b=self.rchild.get_depth()+1
        return a if a>b else b

    def look(self):
        #!!!
        list=self.lchild.look()+self.rchild.look()
        list.append(self)
        return list

    def save(self):
        r1=self.lchild.save()
        r2=self.rchild.save()
        return [self.name,r1,r2]

class F_one_n(F_two):
    def __init__(self,name):
        super().__init__(name)
        self.rchild=t_con()
        self.rchild.parent=self
    
    
    def express(self):
        return self.name+'('+self.lchild.express()+','+self.rchild.express()+')'


### T_node
class t_x(T_node):
    def __init__(self):
        super().__init__(name=random.choice(t_nodes))

    def forward(self,input,n=1):
        if n==1:
            return np.array([input[self.name][-1:][0]/input[self.name][-2:][0]])
        
        l=len(input[self.name])

        if len(input[self.name])>n:
            return input[self.name][l-n:]/input[self.name][l-n-1:-1]
        else:
            return input[self.name][1:]/input[self.name][:-1]
    
class t_con(T_node):
    def __init__(self):
        self.value=np.random.randint(1,15)
        super().__init__(name=str(self.value))
    
    def forward(self,input,n=1):
        if n==1:
            return np.array([self.value])
        else:
            a=[self.value for i in range(2,n+2)]
            return np.array(a)
    
    def update_value(self,value):
        self.value=value
        self.name=str(value)

### F_one
class f_sin(F_one):
    def __init__(self):
        super().__init__(name='sin')
    def forward(self,input,n=1):
        r=self.child.forward(input,n)
        return np.sin(r)
class f_log(F_one):
    def __init__(self):
        super().__init__(name='log')
    def forward(self,input,n=1):
        r=self.child.forward(input,n)
        return np.log(np.abs(r)+1e-8)

class f_exp(F_one):
    def __init__(self):
        super().__init__(name='exp')
    def forward(self,input,n=1):
        r=self.child.forward(input,n)
        return np.exp(np.clip(r,-20,20))

### F_one_n

class f_avg(F_one_n):
    def __init__(self):
        super().__init__(name='avg')
    def forward(self,input,n):
        r1=self.lchild.forward(input,self.rchild.value)
        return np.array([np.average(r1)])


class f_rsi(F_one_n):
    def __init__(self):
        super().__init__(name='rsi')
    def forward(self,input,n):
        r1=self.lchild.forward(input,self.rchild.value)
        # 涨跌拆分
        up = np.maximum(r1, 0) 
        down = np.minimum(r1, 0) 
        # 平均涨跌
        roll_up = np.average(up)
        roll_down = np.average(down)
        RS = roll_up / roll_down if roll_down!=0 else roll_up
        RSI = 100 - (100 / (1 + RS))
        return np.array([RSI])

class f_roc(F_one_n):
    def __init__(self):
        super().__init__(name='roc')
    def forward(self,input,n):
        r1=self.lchild.forward(input,self.rchild.value)
        prices = np.cumprod(1 + r1)
        if prices[0]==0:
            roc=prices[len(prices)-1]*100
        else:
            roc = (prices[len(prices)-1] - prices[0]) / prices[0] * 100
        return np.array([roc])
    
### F_two
class f_add(F_two):
    def __init__(self):
        super().__init__(name='add')
    def forward(self,input,n=1):
        r1=self.lchild.forward(input,n)
        r2=self.rchild.forward(input,n)
        if len(r1)<len(r2):
            r2=r2[-len(r1):]
        if len(r2)<len(r1):
            r1=r1[-len(r2):]
        
        return r1+r2

class f_sub(F_two):
    def __init__(self):
        super().__init__(name='sub')
    def forward(self,input,n=1):
        r1=self.lchild.forward(input,n)
        r2=self.rchild.forward(input,n)
        if len(r1)<len(r2):
            r2=r2[-len(r1):]
        if len(r2)<len(r1):
            r1=r1[-len(r2):]
        return r1-r2

class f_mul(F_two):
    def __init__(self):
        super().__init__(name='mul')
    def forward(self,input,n=1):
        r1=self.lchild.forward(input,n)
        r2=self.rchild.forward(input,n)
        if len(r1)<len(r2):
            r2=r2[-len(r1):]
        if len(r2)<len(r1):
            r1=r1[-len(r2):]
        
        return r1*r2

class f_div(F_two):
    def __init__(self):
        super().__init__(name='div')
    def forward(self,input,n=1):
        r1=self.lchild.forward(input,n)
        r2=self.rchild.forward(input,n)
        if len(r1)<len(r2):
            r2=r2[-len(r1):]
        if len(r2)<len(r1):
            r1=r1[-len(r2):]
        a=[]
        for i in range(len(r1)):
            a.append(r1[i]/r2[i] if r2[i]>0 else r1[i])
        return np.array(a)

# t_list=[t_con,t_x]
# f1_list=[f_abs,f_sin,f_cos,f_log,f_exp]
# f1n_list=[f_avg,f_max,f_min,f_lag,f_momentum,f_rsi,f_roc]
# f2_list=[f_add,f_sub,f_mul,f_div]
# f_list=f1_list+f1n_list+f2_list

t_list=[t_con,t_x]
f1_list=[f_abs,f_sin,f_cos,f_sqrt]
f1n_list=[f_avg,f_max,f_min,f_lag,f_momentum,f_rsi,f_roc]
f2_list=[f_add,f_sub,f_mul,f_div]
f_list=f1_list+f1n_list+f2_list