import os
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# Caminho da pasta com os arquivos CSV
folder_path = 'experimentos/evals'

# Configuração: escolha como tratar valores NaN ('remove' ou 'fill_zero')
na_handling = 'fill_zero'  # Escolha: 'remove' para excluir ou 'fill_zero' para substituir por 0

# Função para carregar todos os CSVs e consolidar os dados
def load_data(folder_path):
    data = []
    for i in range(1, 32):  # Arquivos de eval_1 a eval_31
        file_path = os.path.join(folder_path, f"eval_{i}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep="|")
            df['group'] = f"eval_{i}"  # Adiciona o identificador do grupo
            data.append(df[['answer_relevancy', 'group']])
    return pd.concat(data, ignore_index=True)

# Carrega os dados
data = load_data(folder_path)

# Tratamento de NaN
if na_handling == 'remove':
    data = data.dropna(subset=['answer_relevancy'])
elif na_handling == 'fill_zero':
    data['answer_relevancy'] = data['answer_relevancy'].fillna(0)

# Verifica se há variação em todos os grupos
valid_groups = []
for group in data['group'].unique():
    if data[data['group'] == group]['answer_relevancy'].nunique() > 1:
        valid_groups.append(group)
data = data[data['group'].isin(valid_groups)]

# ANOVA
groups = [data[data['group'] == group]['answer_relevancy'] for group in data['group'].unique()]
anova_result = f_oneway(*groups)
print(f"ANOVA result: F={anova_result.statistic}, p={anova_result.pvalue}")

# Teste post hoc (Tukey HSD)
tukey = pairwise_tukeyhsd(endog=data['answer_relevancy'], groups=data['group'], alpha=0.05)
print(tukey)

# Chama a função para salvar o gráfico
tukey.plot_simultaneous()
output_path='critical_difference_plot.png'
plt.savefig(output_path)
