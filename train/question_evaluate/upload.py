import json
import huggingface_hub
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import argparse
import json
import os
STORAGE_PATH = os.getenv("STORAGE_PATH")
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
print(STORAGE_PATH)
with open('tokens.json', 'r') as f:
    token = json.load(f)['huggingface']
login(token=token)
parser = argparse.ArgumentParser()
parser.add_argument("--repo_name", type=str, default="")
parser.add_argument("--max_score", type=float, default=0.7)
parser.add_argument("--min_score", type=float, default=0.3)
parser.add_argument("--min_diversity", type=float, default=0.0, help="Minimum diversity score threshold (default: 0.0, no filtering)")
parser.add_argument("--experiment_name", type=str, default="Qwen_Qwen3-4B-Base_all")
parser.add_argument("--rollouts_repo_name", type=str, default="", help="Optional repo for rollouts dataset (default: <repo_name>_rollouts)")
args = parser.parse_args()

datas= []
for i in range(8):
    try:
        with open(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json', 'r') as f:
            data = json.load(f)
            datas.extend(data)
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue


for i in range(8):
    try:
        os.remove(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json')
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue

scores = [data['score'] for data in datas]
#  print the distribution of scores
import matplotlib.pyplot as plt
plt.hist(scores, bins=11)
plt.savefig('scores_distribution.png')

#count the number  of score between 0.2 and 0.8 
if not args.repo_name == "":
    # Filter by score and diversity threshold
    filtered_datas = []
    for data in datas:
        score = data.get('score', -1)
        diversity_score = data.get('diversity_score', 0.0)  # Default to 0.0 if not present
        answer = data.get('answer', '')
        
        # Check all conditions
        score_ok = args.min_score <= score <= args.max_score
        diversity_ok = diversity_score >= args.min_diversity
        answer_ok = answer != '' and answer != 'None'
        
        if score_ok and diversity_ok and answer_ok:
            filtered_datas.append({
                'problem': data['question'],
                'answer': answer,
                'score': score,
                'diversity_score': diversity_score
            })
    
    print(f"Filtered {len(filtered_datas)} questions (score: [{args.min_score}, {args.max_score}], diversity >= {args.min_diversity})")
    train_dataset = Dataset.from_list(filtered_datas)
    dataset_dict = {"train": train_dataset}
    config_name = f"{args.experiment_name}"
    dataset = DatasetDict(dataset_dict)
    dataset.push_to_hub(f"{HUGGINGFACENAME}/{args.repo_name}",private=True,config_name=config_name)

    # Upload full rollouts dataset (all questions + all answer rollouts)
    rollouts_repo = args.rollouts_repo_name or f"{args.repo_name}_rollouts"
    rollouts_datas = []
    for data in datas:
        rollouts_datas.append({
            "problem": data.get("question", ""),
            "majority_answer": data.get("answer", ""),
            "answer_rollouts": data.get("results", []),
            "raw_rollouts": data.get("rollouts", []),
            "max_voting_score": data.get("score", -1),
            "diversity_score": data.get("diversity_score", 0.0),
        })
    rollouts_dataset = Dataset.from_list(rollouts_datas)
    rollouts_dataset_dict = {"train": rollouts_dataset}
    rollouts_config_name = f"{args.experiment_name}"
    rollouts_dataset = DatasetDict(rollouts_dataset_dict)
    rollouts_dataset.push_to_hub(f"{HUGGINGFACENAME}/{rollouts_repo}",private=True,config_name=rollouts_config_name)







