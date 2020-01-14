from threading import Thread   #  线程模块
from multiprocessing  import Process,Pool,Queue   # 进程模块
import time
from multiprocessing import Value, Array, Lock


def works(a,d):
    print("aaaaa")
    
def works2(q):  # 多线程调用不能有返回值， 使用Queue存储多线程运算结果
    res=0
    for i in range(10000000):
        res+=i+i**2+i**3
    q.put(res)  # 队列中
    
def normal():
    res=0
    for _ in range(2):
        for i in range(10000000):
            res+=i+i**2+i**3
    print("normal:",res)  # 队列中
    

#创建线程与进程
def cerate_thread():
    start =time.time()
    t1 = Thread(target = works, args=(1,2))
    t1.start()   # 开启线程
    t1.join() # 等待主线程,即守护进程
    print("单线程耗时:%8.f s"%(time.time()-start))
    
def creat_process():
    start = time.time()
    p1 = Process(target =works, args=(1,2))  # args传递两个值时不需要加逗号
    p1.start()
    p1.join()    #等待子-结束偶再继续往下运行，用于进程间同步
    print("单进程耗时:%8.f s"%(time.time()-start))
    
def multi_process():   # 多进程为多核运算  双进程操作
    start = time.time()
    q=Queue()  # 空队列  采用队列的形式进行保存结果
    p1 =Process(target = works2, args=(q,))
    p2 =Process(target = works2, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 =q.get()  # 结果分两批
    res2 =q.get()
    print("多进程结果:",res1+res2)
    print("多进程耗时:%.8f s"%(time.time()-start))
    
def multi_thread():  # 双线程操作
    start = time.time()
    q=Queue()
    t1 =Thread(target = works2,args=(q,))   
    t2 =Thread(target = works2,args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 =q.get()
    res2 =q.get()
    print("多线程结果:",res1+res2)
    print("多线程耗时:%.8f s"%(time.time()- start))
    

def sample_cal():  # 单线程，单进程 对比
    cerate_thread()
    creat_process()
    
    
def multip_cal():  #  多线程， 多进程(多核运算)  对比
    multi_process()
    multi_thread()
    start  = time.time()
    normal()
    print("普通对比函数运行时间：%.8f s"%(time.time()-start))
    
# --------------------进程池-------------------------
# ----------------Pool()----map()-------------------
"""进程池作用: 手动/自动调整进程数量， 让池子对应某一个函数，向池子丢数据，池子返回函数值，
Pool与Process 不同点就是 Pool 函数有返回值，Process 没有返回值
map 用户获取结果， map 中需要放入函数，和需要迭代运算的值，会自动分配给cpu核, 
map 返回多个结果
"""
# --------------apply_async-------------------------
"""
非阻塞操作,apply_async() 只能传递一个值，他会被放入一个核进行计算，传入的值为可迭代，
同时需要get 方法获取返回值
当需要 输出多个值时


"""
def works_pool(x):
    return x*x

def muti_Pool():
    pool=Pool(processes =3)  # 自定义cpu核心数
    res=pool.map(works_pool,range(10))
    print("map测试:",res)
    
    res=pool.apply_async(works_pool,(2,))  # 非阻塞  
    print("apply_async测试:",res.get()) # 使用get 获取结果
    #  返回迭代器，  效果与map 效果一致
    multi_res = [pool.apply_async(works_pool, (i,)) for i in range(10)]
    print("apply_asybc多输出测试:",[res.get() for res in multi_res])
    pool.close()  # 起到阻塞作用， 用于先运行子进程，执行完毕后，回到主进程，以完成下面的操作
    pool.join()
    

# 共享内存
def share_value():
    value1 = Value("i",0)
    value2 = Value("d",3.14)
#    array = Array("i",[1,2,3,4])  # Array 只能是一维的，不能为多维
    return print(value1.value, value2.value)

# --------------------------进程锁--------------------------------------
def works3(v,num):
    for _ in range(5):
        time.sleep(0.1)
        v.value += num  # v.value 获取共享变量值
        print(v.value)

def multicore_unlock():  # 不带进程锁,测试, 存在进程之间抢资源
    v = Value("i",0)  #定义共享变量
    p1 =Process(target = works3, args=(v,1))
    p2 =Process(target = works3, args=(v,3))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


def works4(v,num, l):  # 保证运行时，一个进程对锁内容的独占
    l.acquire()  # 锁住
    for _ in range(5):
        time.sleep(0.1)
        v.value += num  
        print(v.value)
    l.release()  # 释放锁


def multicore_lock():   # 加进程锁，防止冲突
    l =Lock()  # 定义一个进程锁
    v = Value("i",0)  #定义共享变量
    p1 =Process(target = works4, args=(v,1,l))
    p2 =Process(target = works4, args=(v,3,l))
    p1.start()
    p2.start()
    p1.join()
    p2.join()  # 在终端运行出结果

def process_lock_test():
    #  ---------测试带锁与不带锁部分---cmd下进行-----------
    print("不带线程锁")
    multicore_unlock()  # 不带锁进程
    print("带线程锁")
    multicore_lock()  # 带锁进程

if __name__ =="__main__":
#    process_lock_test()   线程锁测试
#    share_value()  # 共享变量测试
#    muti_Pool()  # 线程池测试
#    multip_cal()   # 多线程与多进程耗时对比
    sample_cal()  # 单线程与单进程 对比
    