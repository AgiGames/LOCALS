import pandas as pd
from collections import defaultdict

data = pd.read_csv('seed_data.csv')
mean = data.mean()
std = data.std()

mean_and_std = defaultdict(list)
for column_name in data.columns:
    mean_and_std[column_name].append(f'{mean[column_name]:.3f} ± {std[column_name]:.3f}')
mean_and_std = pd.DataFrame(mean_and_std)
mean_and_std.to_csv('seed_stats.csv', index=False)