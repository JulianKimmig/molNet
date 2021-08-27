import pandas as pd
from multiprocessing import Pool

def split_dataframe(df, chunk_size): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def multi_process_apply(df,call,worker,**apply_kwargs):
    #TODO Multiprocessing
    #worker=max(1,int(worker))
    worker=1
    def f(sdf):
        return sdf.apply(call,**apply_kwargs)
    #print(call,worker)
    if worker == 1:
        return f(df)
    
    with Pool(processes=worker) as pool:
        df = pd.concat(
            pool.map(f,split_dataframe(df,worker))
        )
    return df