import numpy as np
import pandas as pd
from collections import defaultdict

# Parameters
num_annotators = 30
num_models = 5
num_formulas = 15
items_per_person = 15  # 3 items from each model
models_per_person = 3  # Each person sees 3 outputs per model

# Define the models
models = [
    'codellama',
    'mixtral',
    'gemini',
    'gpt',
    'phi'
]

# Generate 75 outputs from 5 models, split them into groups by model
outputs = [f'{model}_Formula{input_id}' for model in models for input_id in range(1, 16)]

# Initialize annotation matrix: rows are annotators, columns are the items assigned
annotation_matrix = np.zeros((num_annotators, items_per_person), dtype=object)

# Distribute the items across annotators
for i in range(num_annotators):
    selected_items = []
    
    # For each model, select 3 items in a round-robin manner
    for model_idx in range(num_models):
        # Calculate the starting index based on the annotator index
        start_index = (model_idx * models_per_person + i) % num_formulas
        # Select 3 items in order, wrapping around if necessary
        for j in range(models_per_person):
            selected_items.append(outputs[model_idx * num_formulas + (start_index + j) % num_formulas])
    
    # Assign the selected 15 items to the annotator
    annotation_matrix[i] = selected_items

# Convert the annotation matrix to a DataFrame for easier visualization
df_annotation_matrix = pd.DataFrame(annotation_matrix, index=[f"Annotator {i+1}" for i in range(num_annotators)],
                                    columns=[f"Item {i+1}" for i in range(items_per_person)])

############################################################################################################

# Initialize dictionaries to track counts
model_counts = defaultdict(int)
annotator_counts = defaultdict(int)
output_counts = defaultdict(int)

# Count annotations per model, annotator, and output
for annotator in df_annotation_matrix.index:
    for item in df_annotation_matrix.loc[annotator]:
        # Count for models
        model = item.split('_')[0]
        model_counts[model] += 1
        
        # Count for annotators
        annotator_counts[annotator] += 1
        
        # Count for specific outputs
        output_counts[item] += 1

# Convert the counts to DataFrames for better visualization
df_model_counts = pd.DataFrame(list(model_counts.items()), columns=['Model', 'Count'])
df_annotator_counts = pd.DataFrame(list(annotator_counts.items()), columns=['Annotator', 'Count'])
df_output_counts = pd.DataFrame(list(output_counts.items()), columns=['Output', 'Count'])

# Display the counts
print("Annotations per Model:")
print(df_model_counts)

print("\nAnnotations per Annotator:")
print(df_annotator_counts)

print("\nAnnotations per Output:")
print(df_output_counts.sort_values('Output')['Count'].tolist())

# Display the annotation matrix
print("\nAnnotation Matrix:")
print(df_annotation_matrix)
df_annotation_matrix.to_csv('data/annotation_matrix.csv')

# Display unique rows in the annotation matrix
unique_rows = df_annotation_matrix.drop_duplicates()
print("\nUnique Rows in Annotation Matrix:")
print(unique_rows)