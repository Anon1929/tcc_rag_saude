import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Caminho para a pasta de resultados de post-hoc
posthoc_folder = "experimentos/posthocs/"

# Listar todos os arquivos na pasta
posthoc_files = [f for f in os.listdir(posthoc_folder) if f.endswith(".csv")]

if not posthoc_files:
    print("Nenhum arquivo de post-hoc encontrado na pasta 'experimentos/posthocs/'.")
else:
    for posthoc_file in posthoc_files:
        file_path = os.path.join(posthoc_folder, posthoc_file)
        
        # Carregar a matriz de p-valores
        nemenyi_results = pd.read_csv(file_path, index_col=0)
        
        # Exibir a matriz no terminal
        print(f"\nMatriz de p-valores - {posthoc_file}:")
        print(nemenyi_results)
        
        # Criar um heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(nemenyi_results, annot=True, fmt=".3f", cmap="coolwarm", cbar=True,
                    xticklabels=nemenyi_results.columns,
                    yticklabels=nemenyi_results.index)
        plt.title(f"Teste Post-hoc Nemenyi - {posthoc_file}")
        plt.xlabel("Configurações")
        plt.ylabel("Configurações")
        plt.tight_layout()
        
        # Salvar o heatmap como PNG
        output_file = os.path.join(posthoc_folder, f"{os.path.splitext(posthoc_file)[0]}_heatmap.png")
        plt.savefig(output_file)
        plt.show()
        
        print(f"Heatmap salvo como '{output_file}'.")
