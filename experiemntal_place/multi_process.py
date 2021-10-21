from multiprocessing import Pool
import multiprocessing as mp
import os
import numpy as np

def f(n):
    return np.var(np.random.sample((n, n)))


if __name__ == '__main__':
    result_objs = []
    n = 100
    with Pool(processes=mp.cpu_count()) as pool:

        result_objs = [pool.apply_async(f, (i+1,)) for i in range(n)]
        
        results = [result.get() for result in result_objs]
        print(len(results), np.mean(results), np.var(results))

    print(results)