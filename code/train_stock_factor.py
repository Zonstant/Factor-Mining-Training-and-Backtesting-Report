from gp_evo import *
from gp_llm import *
from util import *
import time
import matplotlib.pyplot as plt
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData
from vnpy_ctastrategy import (
    CtaTemplate,
    BarData,
    ArrayManager
)
from vnpy_ctastrategy.backtesting import BacktestingEngine
from datetime import datetime
import vnpy
import shutil
import json
from scipy.optimize import curve_fit
from openai import OpenAI
import asyncio
from openai import DefaultAioHttpClient
from openai import AsyncOpenAI

from concurrent.futures import ProcessPoolExecutor

##！！！！关于止盈止损，因为日频没有日内数据，我仅在训练时直接强制其止损，使用分钟频可以策略止损
class Stock_single_Strategy(CtaTemplate):

    author = "YourName"
    parameters = ["tree","kind"]
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.am = ArrayManager(size=15)
        self.tree=tree()
        self.tree.load(setting["tree"])
        self.kind=setting["kind"]
        self.content=[]
        for symbol,exchange,file in get_stock_labels('stock_zz1000_inda'):
            if vt_symbol==f"{symbol}.{exchange.value}":
                df=pd.read_csv('stock_zz1000_inda/'+file)
                df = df.drop(['ts_code','pe_ttm', 'dv_ttm', 'pe', 'dv_ratio','datetime'], axis=1)
                df = df.sort_values("trade_date")
                df = df.set_index("trade_date")
                self.features={}
                for i in df.columns.tolist():
                    self.features[i]=[]
                df["free_share"] = df["free_share"].ffill()
                df["turnover_rate_f"] = df["turnover_rate_f"].ffill()
                df["pb"] = df["pb"].ffill()
                self.df=df
                break

    def on_init(self):
        self.load_bar(15)
    def on_start(self):
        pass
    def on_stop(self):
        pass
    def on_bar(self, bar: BarData):
        self.am.update_bar(bar)
        if int(bar.datetime.date().strftime("%Y%m%d")) not in self.df.index:
            return
        factor = self.df.loc[int(bar.datetime.date().strftime("%Y%m%d"))]
        for i in self.features.keys():
            self.features[i].append(factor[i].item())
        if not self.am.inited:
            return
        close_array = self.am.close

        #tree的输入仅包含历史数据
        if self.kind=='tomorrow':
            input={'price':np.array(close_array[:-1])}
            for i in self.features.keys():
                input[i]=np.array(self.features[i][-15:-1])
            result=self.tree.forward(input)
            self.content.append((result,close_array[-1:][0]/close_array[-2:][0], bar.datetime))
        else:
            input={'price':np.array(close_array[:-5])}
            for i in self.features.keys():
                input[i]=np.array(self.features[i][-15:-5])
            result=self.tree.forward(input)
            if self.kind=='fourdays_avg_std':
                self.content.append((result,np.average(close_array[-4:]/close_array[-5:][0])-np.std(close_array[-4:]/close_array[-5:][0]), bar.datetime))
            elif self.kind=='fourdays_avg':
                self.content.append((result,np.average(close_array[-4:]/close_array[-5:][0]), bar.datetime))
            elif self.kind=='fourdays':
                self.content.append((result,close_array[-1:][0]/close_array[-5:][0], bar.datetime))
                
                
##engine.run_backtesting()占据约90%的时间，其中tree占2/3
def fitnessForstock(treemsg,train_start,train_end,stock,kind):
    symbol,exchange,file=stock
    engine = BacktestingEngine()
    engine.output = lambda msg: None
    engine.set_parameters(
        vt_symbol=f"{symbol}.{exchange.value}",
        interval=vnpy.trader.constant.Interval.DAILY,
        start=train_start,
        end=train_end,
        rate=0.000,
        slippage=0.0,
        size=100,
        pricetick=0.01,
        capital=1000000,
    )
    engine.add_strategy(Stock_single_Strategy, {"tree":treemsg,"kind":str(kind)})
    engine.load_data()
    engine.run_backtesting()
    df = engine.calculate_result()
    engine.calculate_statistics()
    content = engine.strategy.content
    return content

def rank_ic_np(factor, future_return):
    x = np.array(factor)
    y = np.array(future_return)
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    ic = np.corrcoef(x_rank, y_rank)[0,1]
    return ic

def fitness(args):
    try:
        tree,train_start,train_end,stocks,kind=args
        results={}
        for stock in stocks:
            content=fitnessForstock(tree.save(),train_start,train_end,stock,kind)
            for f,price,date in content:
                if date not in results.keys():
                    results[date]=[]
                results[date].append((f,price))
        result_ic=[]
        for date in results.keys():
            a=np.array(results[date])
            result_ic.append(rank_ic_np(a[:,0],a[:,1]).item())
        return (np.abs(np.average(result_ic)/np.std(result_ic)),np.abs(np.average(result_ic)),result_ic)
    except Exception as e:
        print("fitness error:", args)
        return (0,0,[0])
    
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

def gaussian_fit(x,y,size):
    seq=np.argsort(x)
    x=x[seq]
    y=y[seq]
    y/=np.sum(y)
    if len(x)<3:
        return None,None,None
    try:
        params, covariance = curve_fit(
            gaussian,
            x,
            y,
            p0=[np.max(y), np.sum(x*y), 3],
            maxfev=100000
        )
    except RuntimeError:
        print("Gaussian fit failed",x,y)
        return None,None,None
    a, b, c = params
    x_fit=np.array([i for i in range(5,21)])
    y_fit=gaussian(x_fit,a,b,max(1,c))
    y_fit/=np.sum(y_fit)
    sum=0
    p=[]
    for i in y_fit:
        sum+=i
        p.append(sum)
    y_fit*=size
    return x_fit,y_fit,p

def get_p_size(x,p):
    a=random.random()
    for i in range(len(p)):
        if a<p[i]:
            return x[i]

def crossover_count(r1,r2,sizes_depth,sizes_count):
    o1=r1.copy()
    o2=r2.copy()
    n_list1=o1.look()
    n_list2=o2.look()
    f1=False
    f2=False
    count=0
    while (not f1) and (not f2) and count<100:
        count+=1
        n1=random.choice(n_list1)
        n2=random.choice(n_list2)
        if n1.parent is None or n2.parent is None or (isinstance(n1.parent,F_one_n) and n1.parent.rchild==n1) or (isinstance(n2.parent,F_one_n) and n2.parent.rchild==n2):
            continue
        #n2->n1，保留n1
        d1=n1.depth+n2.get_depth()-1
        if d1>=sizes_depth[0] and d1<=sizes_depth[len(sizes_depth)-1] and d1 in sizes_depth:
            if sizes_count[np.where(sizes_depth== d1)[0]]>0:
                f1=True
        #n1->n2，保留n2        
        d2=n2.depth+n1.get_depth()-1
        if d2>=sizes_depth[0] and d2<=sizes_depth[len(sizes_depth)-1] and d2 in sizes_depth:
            if sizes_count[np.where(sizes_depth== d2)[0]]>0:
                f2=True
    if f1:
        p1=n1.parent
        pos1=obtain_pos(n1,p1)
        put_pos(n2,p1,pos1)
    else:
        o1=None
    if f2:
        p2=n2.parent
        pos2=obtain_pos(n2,p2)
        put_pos(n1,p2,pos2)
    else:
        o2=None
    # print(d1,o1.root.get_depth(),f1)
    # print(d2,o2.root.get_depth(),f2)
    return o1,o2



async def request(text,kind) -> None:
    async with AsyncOpenAI(

    ) as client:
        system_text,user_text=generate(kind,kind+".txt",text)
        chat_result = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
        )
        return (chat_result.choices[0].message.content,text)


async def request_all(texts):
    tasks = [request(text,kind) for text,kind in texts]
    results = await asyncio.gather(*tasks)
    return results


def gp(args):


    popSize=50
    generation=5
    k=3
    stocks=get_stock_labels('stock_zz1000')
    stocks=np.array(stocks)


    if args['ai']:
        client = OpenAI(

        )
    

    if os.path.exists(args['out_path']):
        shutil.rmtree(args['out_path'])
    os.makedirs(args['out_path'])

    train_ic=[]
    train_ir=[]
    test_ic=[]
    test_ir=[]
    depths_avg=[]

    #!如果不是maxdepth模式，那么有两个问题：sizecount<0，部分size可能无法到达
    pop=[]
    while(len(pop)<popSize/2):
        t=tree()
        t.full(random.choice([3,4,5]))
        pop.append(t)
        
    while(len(pop)<popSize):
        t=tree()
        t.grow(random.choice([3,4,5]))
        pop.append(t)
    
    #!如果sub_stocks，那么就涉及pop fitness的更新和历史fitness的利用
    
    sub_stocks=stocks[np.random.choice([i for i in range(len(stocks))], 20, replace=False)]
    sub_stocks=stocks[:20]
    popargs=[(tree,args['train_start'],args['train_end'],sub_stocks,args['return']) for tree in pop]
    with ProcessPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(fitness, popargs))
    for i in range(popSize):
        pop[i].fitness=results[i][0]
        pop[i].train_ic=results[i][1]
        pop[i].train_ics=results[i][2]
    
    log=open('log.txt','w')
    depth_fitness_lists=[]
    for g in range(generation):
        repeat=set()

        sizes_depth=np.array([i for i in range(3,21)])
        if args['depth'][0]=='max_depth':
            maxdepth=args['depth'][1]
            sizes_count=np.array([popSize for i in range(5,maxdepth+1)]+[0 for i in range(maxdepth+1,21)])
            a=0
            sizes_p=[]
            for i in range(5,maxdepth+1):
                a+=1.0/(maxdepth+1-5)
                sizes_p.append(a)
            sizes_p+=[0 for i in range(maxdepth+1,21)]

        if args['depth'][0]=='gaussian_fix':
            depth=args['depth'][1]
            sizes_count=gaussian(sizes_depth,1,depth,2)
            sizes_count/=np.sum(sizes_count)
            a=0
            sizes_p=[]
            for i in sizes_count:
                a+=i
                sizes_p.append(a)
            sizes_count*=popSize
        
        print(g)
        print(sizes_depth,np.round(sizes_count),np.sum(sizes_count))
        
        for i in pop:
            i.update_depth()
        off=[]

        #每代加入一些随机
        while(len(off)<popSize/20):
            t=tree()
            t.full(get_p_size(sizes_depth,sizes_p))
            if t.express() not in repeat:
                sizes_count[np.where(sizes_depth== t.root.get_depth())[0]]-=1
                off.append(t)
                repeat.add(t.express())
        while(len(off)<popSize/10):
            t=tree()
            t.grow(get_p_size(sizes_depth,sizes_p))
            if t.express() not in repeat:
                sizes_count[np.where(sizes_depth== t.root.get_depth())[0]]-=1
                off.append(t)
                repeat.add(t.express())
        while(len(off)<popSize/5):
            t=k_tournament(pop,k)
            if t.express() not in repeat:
                sizes_count[np.where(sizes_depth== t.root.get_depth())[0]]-=1
                off.append(t)
                repeat.add(t.express())
        
        crossover_sample=[]
        crossover_error=""
        mutation_sample=[]
        mutation_error=""

        #LLM 变异
        if args['ai']:
            texts=[]
            parents=[]
            for i in range(popSize):
                if random.random()<0.1:
                    parents.append(pop[i])
                    texts.append((mutation_text(pop[i],None),'mutation'))
            results=asyncio.run(request_all(texts))
            for i in range(len(texts)):
                result,text=results[i]
                if not ('<' in result and '>' in result):
                    mutation_error+=text[:-1]
                    mutation_error+=","+result[:-1]+", violate the required output format.}],"
                else:
                    t=postfix_to_tree(result[result.find('<')+1:result.find('>')])
                    if t is None:
                        mutation_error+=text[:-1]
                        mutation_error+=","+result[:-1]+", invalid tree.}],"
                    elif t =='invalid depth':
                        mutation_error+=text[:-1]
                        mutation_error+=","+result[:-1]+", invalid depth.}],"
                    else:
                        off.append(t)
                        mutation_sample.append((parents[i],t))
        
        #!!!!!检查max_depth不一样
        if args['ai']:
            tar_num=(popSize+len(off))/2
        else:
            tar_num=popSize
        while(len(off)<tar_num):
            r1=k_tournament(pop,k)
            r2=k_tournament(pop,k)
            o1,o2=crossover_count(r1,r2,sizes_depth,sizes_count)
            if o1 is not None and o1.express() not in repeat and o1.root.get_depth() in sizes_depth:
                index=np.where(sizes_depth== o1.root.get_depth())[0]
                if sizes_count[index]>0:
                    sizes_count[index]-=1
                    off.append(o1)
            if o2 is not None and o2.express() not in repeat and o2.root.get_depth() in sizes_depth:
                index=np.where(sizes_depth== o2.root.get_depth())[0]
                if sizes_count[index]>0:
                    sizes_count[index]-=1
                    off.append(o2)
        
        if args['ai']:
            #加入一半的LLM结果，如果不想加，则要把上述popSize/2改为popSize
            while(len(off)<popSize):
                texts=[]
                parents=[]
                for i in range(popSize-len(off)):
                    r1=k_tournament(pop,k)
                    r2=k_tournament(pop,k)
                    parents.append((r1,r2))
                    texts.append((crossover_text(r1,r2,None),'crossover'))
                results=asyncio.run(request_all(texts))
                for i in range(len(texts)):
                    result,text=results[i]
                    if not ('<' in result and '>' in result):
                        crossover_error+=text[:-1]
                        crossover_error+=","+result[:-1]+", violate the required output format.}],"
                    else:
                        t=postfix_to_tree(result[result.find('<')+1:result.find('>')])
                        if t is None:
                            crossover_error+=text[:-1]
                            crossover_error+=","+result[:-1]+", invalid tree.}],"
                        elif t =='invalid depth':
                            crossover_error+=text[:-1]
                            crossover_error+=","+result[:-1]+", invalid depth.}],"
                        else:
                            off.append(t)
                            crossover_sample.append((parents[i][0],parents[i][1],t))
                print(len(off))
        
        if not args['ai']:
            for i in range(len(off)):
                if random.random()<0.1:
                    t=None
                    while t is None:
                        t= mutation(off[i],off[i].root.get_depth())
                    off[i]=t
        
        #多线程
        popargs=[(tree,args['train_start'],args['train_end'],sub_stocks,args['return']) for tree in off]
        with ProcessPoolExecutor(max_workers=15) as executor:
            results = list(executor.map(fitness, popargs))
        for i in range(len(off)):
            off[i].fitness=results[i][0]
            off[i].train_ic=results[i][1]
            off[i].train_ics=results[i][2]
        


        if args['ai']:
            pop_sorted = sorted(crossover_sample, key=lambda x: (x[0].fitness+x[1].fitness)/2-x[2].fitness)
            n = len(pop_sorted)
            result = []
            for i in range(5):
                start = int(i * n / 5)
                end = int((i + 1) * n / 5)
                layer = pop_sorted[start:end]
                if len(layer) <= 10:
                    result.append(layer)
                else:
                    result.append(random.sample(layer, 10))
            text="The individuals are divided into five fitness levels based on the improvement of the child compared to its parents, ordered from best to worst. I will provide these levels sequentially. "
            for i in range(5):
                text+=f"The {i+1}-the level: "+"{"
                for p1,p2,child in result[i]:
                    text+=crossover_text(p1,p2,child)+","
                text=text[:-1]+"}"
            if len(crossover_error)!=0:
                text+="The invalid trees: {"+crossover_error
            text=text[:-1]+"}"
            text+="You should analyze the results by considering both the information of each level and the results within each level. Identify why some expressions perform well and why others perform poorly, and extract common patterns across different levels."
            log.write(str(g)+'\n')
            log.write('crossover\n')
            log.write(text+'\n')
            system_text,user_text=think("crossover","crossover.txt",text)
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                stream=False
            )
            with open("crossover.txt", "w", encoding="utf-8") as f:
                f.write(response.choices[0].message.content)
                f.close()
            
        
        if args['selection']==0:
            pop=pop+off
            pop=selection(pop,popSize,k)
        else:
            pop=off
        
        depths=[(i.root.get_depth(),i.fitness) for i in pop]
        depth_fitness_lists+=depths
        depths_avg.append(np.average(np.array(depths)[:,0]))

        if args['depth'][0]=='gaussian_update':
            last_depth=sizes_depth.copy()
            last_count=sizes_count.copy()
            last_p=sizes_p.copy()

            if args['depth'][1]=='current':
                elite=sorted(depths,key=lambda x:-1*x[1])
            elif args['depth'][1]=='pop':
                elite=sorted(depth_fitness_lists,key=lambda x:-1*x[1])
            else:
                print('error depth')
            
            if args['depth'][2]=='best':
                elite=np.array(elite)[:(int)(len(elite)/3)]
            elif args['depth'][2]=='best_partial':
                elite=np.array(elite)[(int)(len(elite)/20):(int)(len(elite)/3)]
            elif args['depth'][2]=='best_fix':
                elite=np.array(elite)[:1000]
                elite=np.array(elite)[(g+1)*10:1000+(g+1)*10]
            else:
                print('error depth')
            
            if args['depth'][3]=='count':
                x,y= np.unique(elite[:,0], return_counts=True)
            elif args['depth'][3]=='ftness_total':
                x,y= np.unique(elite[:,0], return_counts=True)
                x=x.astype(int)
                tem_y=[0 for i in range(np.max(x)+1)]
                for a,b in elite:
                    tem_y[int(a)]+=b
                y=[tem_y[a]  for a in x]
            elif args['depth'][3]=='fitness_avg':
                x,y= np.unique(elite[:,0], return_counts=True)
                x=x.astype(int)
                tem_y=[0 for i in range(np.max(x)+1)]
                for a,b in elite:
                    tem_y[int(a)]+=b
                y=np.array([tem_y[a]  for a in x])/y
            else:
                print('error depth')
            
            sizes_depth,sizes_count,sizes_p=gaussian_fit(np.array(x),np.array(y,dtype=float),popSize)
            if sizes_depth is None:
                sizes_depth,sizes_count,sizes_p=last_depth,last_count,last_p

        #for each size, calculate the score of better ones
        if args['depth'][0]=='update_size':
            tem_y=[[] for i in range(int(np.max(np.array(depth_fitness_lists)[:,0])+1))]
            for a,b in depth_fitness_lists:
                tem_y[int(a)].append(b)
            sizes_depth=[]
            sizes_count=[]
            for i in range(5,21):
                sizes_depth.append(i)
                yy=sorted(tem_y[i])
                yy=yy[-(int)(len(yy)/10):]
                if(len(yy)>0):
                    sizes_count.append(np.average(yy))
                else:
                    sizes_count.append(0)
            sizes_depth=np.array(sizes_depth)
            sizes_count=np.array(sizes_count)
            sizes_count/=np.sum(sizes_count)
            sizes_count*=popSize
            sizes_count=np.maximum(sizes_count,20)
            sizes_count/=np.sum(sizes_count)
            a=0
            sizes_p=[]
            for i in sizes_count:
                a+=i
                sizes_p.append(a)
            sizes_count*=popSize
        
        #验证集
        popargs=[(tree,args['test_start'],args['test_end'],sub_stocks,args['return']) for tree in pop]
        with ProcessPoolExecutor(max_workers=15) as executor:
            results = list(executor.map(fitness, popargs))
        for i in range(popSize):
            pop[i].test_ir=results[i][0]
            pop[i].test_ic=results[i][1]
        
        elite=sorted(pop,key=lambda x:-1*x.fitness)[:int(popSize/10)]
        train_ic.append(np.average([t.train_ic for t in elite]))
        train_ir.append(np.average([t.fitness for t in elite]))
        test_ic.append(np.average([t.test_ic for t in elite]))
        test_ir.append(np.average([t.test_ir for t in elite]))

        with open(args['out_path']+'/'+str(g+1)+".json", "w") as f:
            json.dump([i.save() for i in pop], f)


        fig, ax1 = plt.subplots()
        l11=ax1.plot([i for i in range(len(train_ic))], train_ic, label="train_ic",color='blue')
        l12=ax1.plot([i for i in range(len(train_ir))], train_ir, label="train_ir",color='black')
        l13=ax1.plot([i for i in range(len(test_ic))], test_ic, label="test_ic",color='green')
        l14=ax1.plot([i for i in range(len(test_ir))], test_ir, label="test_ir",color='orange')
        ax1.set_ylabel("value")
        plt.grid()
        ax2 = ax1.twinx()
        l2=ax2.plot([i for i in range(len(depths_avg))], depths_avg, label="avg",color='red',linestyle='--')
        ax2.set_ylabel("size")
        plt.xlabel("epoch")
        lines = l11+l12+l13+l14+l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels)
        plt.title(args['name'])
        plt.savefig(args['name']+".png")
        plt.close(fig)
    log.close()

if __name__=='__main__':

    args={}
    args['train_start']=datetime(2025, 4, 1)
    args['train_end']=datetime(2025, 10, 1)
    args['test_start']=datetime(2025, 10, 2)
    args['test_end']=datetime(2026, 1, 1)
    args['selection']=1
    #量价:price 基本面: feature, all:all
    #请直接更改t_nodes
    args['feature']='all'
    #预测什么return:'tomorrow','fourdays_avg_std','fourdays_avg','fourdays'
    args['return']='fourdays_avg_std'
    
    args['depth']=['max_depth',12]
    #args['depth']=['gaussian_fix',12]

    #url请不要复制，换成自己的
    args['ai']=False

    args['name']='newstock_factor_'+str(args['return'])+'_'+str(args['selection'])
    args['out_path']='./result_'+str(args['return'])+'_'+str(args['selection'])

    gp(args)