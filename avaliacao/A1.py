import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from sklearn.preprocessing import StandardScaler
from scikit_posthocs import critical_difference_diagram

# Caminho para a pasta de experimentos
experiment_folder = "experimentos/evals"

# Lista para armazenar os dados de cada configuração
data = []
metricas = ["faithfulness","answer_relevancy","context_recall","context_precision","semantic_similarity","answer_correctness"]

# Carregar os dados de cada configuração
for metric in metricas:

    dfs = []
    for i in range(1, 31):
        file_path = os.path.join(experiment_folder, f"eval_{i}.csv")
        # Carregar o arquivo CSV e pegar a coluna 'context_precision'
        df = pd.read_csv(file_path, sep='|', usecols=[metric])
        # Adicionar uma coluna no DataFrame com o nome do arquivo
        df.columns = [f'eval_{i}']
        
        df.fillna(0, inplace=True)

        # Adicionar o DataFrame à lista
        
        dfs.append(df)

    # Concatenar todos os DataFrames horizontalmente
    consolidated_df = pd.concat(dfs, axis=1)

    # Exibir o DataFrame consolidado
    consolidated_df.to_csv("metricasmatrix/"+metric+".csv")