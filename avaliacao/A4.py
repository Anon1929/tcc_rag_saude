import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikit_posthocs as sp

folder_path = 'experimentos/evals'

na_handling = 'fill_zero' 

def load_data(folder_path):
    data = []
    for i in range(1, 32):  
        file_path = os.path.join(folder_path, f"eval_{i}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep="|")
            data.append(df['answer_relevancy'])
    return pd.concat(data, axis=1, ignore_index=True)


data = load_data(folder_path)


if na_handling == 'remove':
    data = data.dropna(axis=0) 
elif na_handling == 'fill_zero':
    data = data.fillna(0)  

data.columns = [f"eval_{i}" for i in range(1, data.shape[1] + 1)]


from scipy.stats import friedmanchisquare
friedman_stat, friedman_p = friedmanchisquare(*[data[col] for col in data.columns])
print(f"Friedman test: χ2={friedman_stat}, p={friedman_p}")

if friedman_p < 0.05:
    nemenyi_result = sp.posthoc_nemenyi_friedman(data.to_numpy())

    def save_critical_difference_diagram(data, output_path='nemenyi_cd_plot.png'):
        labels = data.columns.tolist()
        sp.sign_plot(nemenyi_result, labels=labels, alpha=0.05)
        plt.title('Critical Difference Diagram (Nemenyi Test)')
        plt.savefig(output_path)
        print(f"Critical difference diagram saved as: {output_path}")
    print(nemenyi_result)
    save_critical_difference_diagram(nemenyi_result)

else:
    print("Friedman test não significante.")
