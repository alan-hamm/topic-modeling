

#%%

import matplotlib.pyplot as plt
import pandas as pd
import os 

START_TOPICS = 50
END_TOPICS = 61
STEP_SIZE = 1

train_metrics_csv = pd.read_parquet(r"C:\_harvester\data\lda-models\2010s_html\metadata\metadata.parquet")

#%%


print(train_metrics_csv.head())



#%%

log_dir = r'C:\_harvester\data\lda-models\2010s_html\log'

subset_df = train_metrics_csv[train_metrics_csv['type'] == 'eval']
# Plotting the performance curves
x_axis=[]
#iterations = range(START_TOPICS, END_TOPICS + 1, STEP_SIZE)
for i in range(len(subset_df)):
    x_axis.append(i)
# Plotting the performance curves for training data
plt.figure(figsize=(25, 15))
plt.plot(x_axis, subset_df['coherence_mean'], label='Coherence Mean')
#plt.plot(iterations, train_metrics_csv['convergence_score'], label='Training Convergence Score')
#plt.plot(iterations, train_metrics_csv['log_perplexity'], label='Training Log Perplexity')

train_data_performance_curve = os.path.join(log_dir, f"training_visual.png")
plt.xlabel('Number of Topics')
plt.ylabel('Metric Value')
plt.title('Training Data | Evaluation Metrics Comparison')
plt.legend()
plt.savefig(train_data_performance_curve)  # Save the figure as an image file
plt.show()

# Plotting the performance curves for evaluation data
#plt.figure(figsize=(10, 5))
#plt.plot(iterations, eval_metrics_csv['cv_score'], label='Evaluation Coherence Score')
#plt.plot(iterations, eval_metrics_csv['convergence_score'], label='Evaluation Convergence Score')
#plt.plot(iterations, eval_metrics_csv['log_perplexity'], label='Evaluation Log Perplexity')

#eval_data_performance_curve = os.path.join(log_dir, f"evaluation_visual.png")
#plt.xlabel('Number of Topics')
#plt.ylabel('Metric Value')
#plt.title('Evaluation Data | Evaluation Metrics Comparison')
#plt.legend()
#plt.savefig(eval_data_performance_curve)  # Save the figure as an image file
#plt.show()
# %%
