import json
import os
import pandas as pd
from collections import defaultdict

all_metrics = defaultdict(list)
project_root = os.path.dirname(os.path.abspath(__file__))
metric_json_file_names = [file_name for file_name in os.listdir(project_root) if file_name.endswith('.json')]

for mjfn in metric_json_file_names:
    with open(mjfn, 'r') as f:
        metrics = json.load(f)
        
    train_split = metrics['train_split']
    test_split = val_split = metrics['test_split']
    seed = metrics['seed']
    
    classification_performance = metrics['classification_performance']
    recall = classification_performance['recall']
    precision = classification_performance['precision']
    f1_score = classification_performance['f1_score']
    
    overall_performance = metrics['overall_performance']
    mean_average_precision = overall_performance['mean_average_precision']
    mean_correlation_score = overall_performance['mean_correlation_score']
    
    all_metrics['train_split'].append(train_split)
    all_metrics['test_split'].append(test_split)
    all_metrics['val_split'].append(test_split)
    all_metrics['seed'].append(seed)
    
    all_metrics['recall'].append(recall)
    all_metrics['precision'].append(precision)
    all_metrics['f1_score'].append(f1_score)
    
    all_metrics['mean_average_precision'].append(mean_average_precision)
    all_metrics['mean_correlation_score'].append(mean_correlation_score)
    
all_metrics_df = pd.DataFrame(all_metrics)
all_metrics_df.to_csv('all_metrics.csv', index=False)

mean_std = all_metrics_df.apply(
    lambda x: f"{x.mean():.3f} ± {x.std():.3f}"
).to_frame().T
mean_std.to_csv("mean_std.csv", index=False)