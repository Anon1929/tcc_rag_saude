import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

input_csv = 'resultRagasEval.csv'  
base_folder = 'ResultExperimentos'  
experiment_prefix = 'Result'

def get_next_experiment_folder(base, prefix):
    n = 0
    while os.path.exists(os.path.join(base, f"{prefix}_{n}")):
        n += 1
    return os.path.join(base, f"{prefix}_{n}")


#Copia arquivos e faz pastas
os.makedirs(base_folder, exist_ok=True)
experiment_folder = get_next_experiment_folder(base_folder, experiment_prefix)
os.makedirs(experiment_folder)
shutil.copy(input_csv, experiment_folder)

output_folder_all = os.path.join(experiment_folder, 'graphs')
output_folder_no_zeros = os.path.join(experiment_folder, 'graphs_no_zeros')
os.makedirs(output_folder_all, exist_ok=True)
os.makedirs(output_folder_no_zeros, exist_ok=True)

df = pd.read_csv(input_csv, sep='|')

# --- Pie Chart for "retrieved_contexts" ---
retrieved_counts = df['retrieved_contexts'].apply(lambda x: x == '[]').value_counts()

# Plot pie chart for "all values" case
plt.figure(figsize=(8, 8))
retrieved_counts.plot.pie(
    labels=['Not "[]"', '"[]"'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['#66c2a5', '#fc8d62']
)
plt.title('Proportion of "retrieved_contexts" Values')
plt.ylabel('')
plt.savefig(os.path.join(output_folder_all, 'retrieved_contexts_pie.png'))
plt.close()

# --- Box Plots for Numeric Columns ---
numeric_columns = [
    'faithfulness',
    'answer_relevancy',
    'context_recall',
    'context_precision',
    'context_entity_recall',
    'semantic_similarity',
    'answer_correctness'
]

# Function to create box plots and save statistics
def create_box_plots(dataframe, folder, exclude_zero=False):
    stats_file = os.path.join(folder, 'boxplot_statistics.txt')
    with open(stats_file, 'w') as f:
        for column in numeric_columns:
            # Count empty (NaN) values
            empty_count = dataframe[column].isna().sum()

            # Prepare column data
            column_data = dataframe[column].dropna()
            if exclude_zero:
                column_data = column_data[column_data != 0]  # Exclude zeros

            # Skip if no valid data
            if column_data.empty:
                continue

            # Calculate statistics
            mean_value = column_data.mean()
            median_value = column_data.median()

            # Write statistics to file
            f.write(f"Column '{column}' ({'excluding zeros' if exclude_zero else 'all values'}):\n")
            f.write(f" - Empty values: {empty_count}\n")
            f.write(f" - Valid values: {len(column_data)}\n")
            f.write(f" - Mean: {mean_value:.2f}\n")
            f.write(f" - Median: {median_value:.2f}\n\n")

            # Plot box plot
            plt.figure(figsize=(6, 8))
            plt.boxplot(column_data, vert=True, patch_artist=True, boxprops=dict(facecolor='#66c2a5'))
            plt.title(f'Box Plot of {column} {"(No Zeros)" if exclude_zero else "(All Values)"}')
            plt.ylabel(column)
            filename = f'{column}_boxplot{"_no_zeros" if exclude_zero else ""}.png'
            plt.savefig(os.path.join(folder, filename))
            plt.close()

create_box_plots(df, output_folder_all, exclude_zero=False)
create_box_plots(df, output_folder_no_zeros, exclude_zero=True)

print(f"Graphs and statistics saved in:\n- {output_folder_all}\n- {output_folder_no_zeros}")
print(f"CSV file copied to {experiment_folder}")
