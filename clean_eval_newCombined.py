import pandas as pd
import json
import os

data_df = pd.read_csv('/cbica/home/xjia/qlora/data/lab/new_combined_qa_pairs_instruction.csv')
eval_data = data_df.iloc[:2000].sample(n=100, random_state=42)
eval_data.sort_index(inplace=True)
output_path = '/cbica/home/xjia/qlora/data/lab/newCombined_bench.jsonl'

for index, row in eval_data.iterrows():
    with open(output_path, 'a') as fout:
        json.dump({
            "question_id": index+1,
            "category": "generic",
            "turns": [row["input"]]
        }, fout)
        fout.write('\n')

print("done")