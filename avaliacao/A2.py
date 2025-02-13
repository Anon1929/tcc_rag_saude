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

# Carregar os dados de cada configuração
for i in range(1, 31):
    file_path = os.path.join(experiment_folder, f"eval_{i}.csv")
    df = pd.read_csv(file_path, sep="|")
    
    # Substituir NaN por 0 na coluna 'context_precision'
    df['context_precision'].fillna(0, inplace=True)
    data.append(df['context_precision'].values)

# Converter para matriz numpy (200 perguntas x 30 configurações)
data_matrix = np.array(data).T

# Filtrar linhas com valores constantes
valid_rows = [row for row in data_matrix if len(set(row)) > 1]
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

    # Verificar diferenças significativas
    if p_value < 0.05:
        print("Diferenças significativas detectadas. Realizando teste post-hoc (Dunn)...")
        
        # Teste de Dunn como post-hoc
        dunn_results = sp.posthoc_dunn(valid_data_matrix, p_adjust='holm')

        # Calcular as médias de ranking para cada configuração
        rankings = np.mean(np.argsort(np.argsort(-valid_data_matrix, axis=0), axis=0), axis=1)

        # Configurar o gráfico de diferença crítica
        plt.figure(figsize=(10, 6))
        critical_difference_diagram(rankings, dunn_results)
        plt.title("Gráfico de Diferença Crítica (CD)")
        cd_diagram_file = "critical_difference_diagram.png"
        plt.savefig(cd_diagram_file)
        plt.show()
        print(f"Gráfico de diferença crítica salvo como '{cd_diagram_file}'.")
    else:
        print("Não há diferenças significativas entre as configurações.")
