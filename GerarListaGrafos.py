import pandas as pd
import matplotlib.pyplot as plt
import os

# Base paths
input_folder = 'experimentos/evals'
output_base_folder = 'experimentos/grafos'
os.makedirs(output_base_folder, exist_ok=True)

# Numeric columns for processing
numeric_columns = [
    'faithfulness',
    'answer_relevancy',
    'context_recall',
    'context_precision',
    'semantic_similarity',
    'answer_correctness'
]

# Summary tables
summary_means_all = []
summary_means_no_zeros = []

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

# Loop through eval files and process them
for i in range(1, 32):
    input_csv = os.path.join(input_folder, f'eval_{i}.csv')
    if not os.path.exists(input_csv):
        print(f"File {input_csv} not found. Skipping.")
        continue

    print(f"Processing {input_csv}...")

    # Load the CSV file
    df = pd.read_csv(input_csv, sep='|')

    # Calculate mean values for the experiment (all values)
    means_all = df[numeric_columns].mean().to_dict()
    means_all['experiment'] = f'eval_{i}'
    summary_means_all.append(means_all)

    # Calculate mean values excluding zeros
    filtered_df = df[numeric_columns].replace(0, pd.NA).dropna()
    means_no_zeros = filtered_df.mean().to_dict()
    means_no_zeros['experiment'] = f'eval_{i}'
    summary_means_no_zeros.append(means_no_zeros)

    # Prepare output folders
    output_folder_all = os.path.join(output_base_folder, f'graphs_{i}', 'all_values')
    output_folder_no_zeros = os.path.join(output_base_folder, f'graphs_{i}', 'no_zeros')
    os.makedirs(output_folder_all, exist_ok=True)
    os.makedirs(output_folder_no_zeros, exist_ok=True)

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

    # Create box plots and save statistics
    create_box_plots(df, output_folder_all, exclude_zero=False)
    create_box_plots(df, output_folder_no_zeros, exclude_zero=True)

# Create summary DataFrames
summary_df_all = pd.DataFrame(summary_means_all)
summary_df_all = summary_df_all[['experiment'] + numeric_columns]  # Ensure experiment column comes first
summary_df_no_zeros = pd.DataFrame(summary_means_no_zeros)
summary_df_no_zeros = summary_df_no_zeros[['experiment'] + numeric_columns]  # Ensure experiment column comes first

# Save summaries to CSV
summary_csv_all = os.path.join(output_base_folder, 'summary_means_all_values.csv')
summary_df_all.to_csv(summary_csv_all, index=False)

summary_csv_no_zeros = os.path.join(output_base_folder, 'summary_means_no_zeros.csv')
summary_df_no_zeros.to_csv(summary_csv_no_zeros, index=False)

# Sort by mean values and save
sorted_summary_all = summary_df_all.sort_values(by=numeric_columns, ascending=False)
sorted_csv_all = os.path.join(output_base_folder, 'summary_sorted_all_values.csv')
sorted_summary_all.to_csv(sorted_csv_all, index=False)

sorted_summary_no_zeros = summary_df_no_zeros.sort_values(by=numeric_columns, ascending=False)
sorted_csv_no_zeros = os.path.join(output_base_folder, 'summary_sorted_no_zeros.csv')
sorted_summary_no_zeros.to_csv(sorted_csv_no_zeros, index=False)

print(f"Graphs and statistics saved in '{output_base_folder}'.")
print(f"Summary tables saved as:\n- {summary_csv_all}\n- {summary_csv_no_zeros}\n- {sorted_csv_all}\n- {sorted_csv_no_zeros}")
