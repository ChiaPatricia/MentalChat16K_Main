import pandas as pd
import os

data_dir = '/cbica/home/xjia/qlora/data/eval_jsonl/JT_bench'
models_response = os.listdir(data_dir)
questions = pd.read_json('/cbica/home/xjia/qlora/data/lab/JT_bench.jsonl', lines=True)

result_df = pd.DataFrame({})
models = []
accs = []

for model_response in models_response:
    if model_response.endswith('.jsonl'):
        model_name = model_response[:-6]
        models.append(model_name)

        model_response_df = pd.read_json(os.path.join(data_dir, model_response), lines=True)
        
        gt = questions['answer']
        preds = model_response_df['predicted_answer']
        acc = (gt == preds).mean()
        accs.append(acc)

result_df['Model'] = models
result_df['Accuracy'] = accs

result_df.to_csv('/cbica/home/xjia/qlora/data/eval_jsonl/JT_bench/acc_result.csv', index=False)

print("done")
