import numpy as np
import pandas as pd
df = []
r_dtypes = {"stars": np.float16, 
            "useful": np.int32, 
            "funny": np.int32,
            "cool": np.int32,
           }

with open("C:/Users/nican/Desktop/IAPROJECT/yelp2021/yelp_academic_dataset_review.json", "r",encoding='utf-8') as f:
    reader = pd.read_json(f, orient="records", lines=True, 
                          dtype=r_dtypes, chunksize=1000)
        
    for chunk in reader:
        reduced_chunk = chunk.drop(columns=['review_id', 'user_id','funny','cool','useful',])\
                             .query("`date` >= '2018-12-31'")#2018/2019
        df.append(reduced_chunk)
     
df = pd.concat(df, ignore_index=True)

with open('YelpReviewReduced.json', 'w') as f:
    f.write(df.to_json(orient='records', lines=True))