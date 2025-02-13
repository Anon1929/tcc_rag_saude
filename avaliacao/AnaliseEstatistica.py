import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from sklearn.preprocessing import StandardScaler

# Caminho para a pasta de experimentos
experiment_folder = "experimentos/evals"

# Lista para armazenar os dados de cada configuração
data = []

# Carregar os dados de cada configuração
for i in range(1, 31):
    file_path = os.path.join(experiment_folder, f"eval_{i}.csv")
    df = pd.read_csv(file_path, sep="|")
    
    # Substituir NaN por 0 na coluna 'context_precision'
    df['context_precision'].fillna(0, inplace=True)
    
    # Exibir estatísticas descritivas para cada configuração
    print(f"Configuração {i} - Estatísticas:")
    print(df['context_precision'].describe())
    print()
    
    data.append(df['context_precision'].values)

# Converter para matriz numpy (200 perguntas x 30 configurações)
data_matrix = np.array(data).T  # Transpor para alinhar perguntas como linhas

# Verificar linhas constantes (se necessário)
valid_rows = []
for row in data_matrix:
    if len(set(row)) > 1:  # Excluir linhas onde todos os valores são iguais
        valid_rows.append(row)

# Converter para numpy array novamente
valid_data_matrix = np.array(valid_rows)

if valid_data_matrix.shape[0] == 0:
    print("Todos os dados foram invalidados por serem constantes.")
else:
    # Padronizar os dados
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(valid_data_matrix)
    
    # Realizar o teste de Friedman
    friedman_stat, p_value = friedmanchisquare(*standardized_data.T)
    print(f"Estatística de Friedman: {friedman_stat:.2f}, p-value: {p_value:.4f}")

    # Verificar se há diferenças significativas
    if p_value < 0.05:
        print("Diferenças significativas entre configurações. Realizando teste post-hoc...")

        # Teste de Dunn como post-hoc
        dunn_results = sp.posthoc_dunn(valid_data_matrix, p_adjust='holm')

        # Salvar resultados do post-hoc em um arquivo
        dunn_file = "posthoc_dunn_results.csv"
        dunn_results.to_csv(dunn_file)
        print(f"Resultados do teste de Dunn salvos em '{dunn_file}'.")
        
        # Visualização: Heatmap
        plt.figure(figsize=(30, 20))
        sns.heatmap(dunn_results, annot=True, fmt=".3f", cmap="coolwarm", cbar=True,
                    xticklabels=[f"C{i}" for i in range(1, 31)],
                    yticklabels=[f"C{i}" for i in range(1, 31)])
        plt.title("Teste Post-hoc Dunn - Matriz de p-valores")
        plt.xlabel("Configurações")
        plt.ylabel("Configurações")
        plt.tight_layout()
        heatmap_file = "posthoc_dunn_heatmap.png"
        plt.savefig(heatmap_file)
        plt.show()
        print(f"Heatmap salvo como '{heatmap_file}'.")
    else:
        print("Não há diferenças significativas entre as configurações.")
