import krippendorff
import numpy as np
import pandas as pd
import plotly.express as px
import re


# Extract the numeric part of each column name to sort, ignoring non-numeric suffixes
def extract_key(col_name):
    # Split at '.' and take the first part for primary sorting
    parts = col_name[1:].split('.')  # Skip 'Q'
    primary = int(parts[0])  # Convert the first part to an int
    secondary = int(parts[1].split('_')[0]) if len(parts) > 1 else 0  # Convert the second part to an int, or use 0
    return primary, secondary

# Cast to numeric
def cast_str_to_int(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            contains_number = df[column].apply(lambda x: bool(re.search(r'^\d+$', str(x))))
            if contains_number.all():
                df[column] = df[column].astype(int)
    return df


# Load the gold data
data_gold = pd.read_csv("evaluation/gold_label/edu_kees.csv")
data_gold = data_gold[data_gold['Progress'] == "100"]
data_gold = data_gold[data_gold['Q1.2'] == "Yes"]

# Keep questions columns
data_gold = data_gold[[col for col in data_gold.columns if col.startswith("Q")]].iloc[:, 2:]
q_columns_sorted = sorted(data_gold, key=extract_key)

ordered_df_gold = data_gold[q_columns_sorted]

data_gold = cast_str_to_int(ordered_df_gold)

# map categories to numbers (keeping numbering as trivago experiment for consistency)
categories = {
    "0a: Well-matched": 1,
    "1a: Output too weak": 3,
    "2a: Output too strong": 5,
    "2b: Output contradictory": 6,
    "3a: Input and Output independent": 7,
    "3b: Input and Output contradictory": 8,
    "Undefined": 9,
}

# reverse mapping
categories_reverse = {v: k for k, v in categories.items()}

# map categories to per step answers
# .3, .4 ambiguity
per_step = {
    1: {"Q{item}.5": "No", "Q{item}.6": "Yes", "Q{item}.7": "Yes"},
    3: {"Q{item}.5": "No", "Q{item}.6": "Yes", "Q{item}.7": "No"},
    5: {"Q{item}.5": "No", "Q{item}.6": "No", "Q{item}.8": "Yes"},
    6: {"Q{item}.5": "Yes"},
    7: {"Q{item}.5": "No", "Q{item}.6": "No", "Q{item}.8": "No", "Q{item}.9": "No"},
    8: {"Q{item}.5": "No", "Q{item}.6": "No", "Q{item}.8": "No", "Q{item}.9": "Yes"},
    9: {"Q{item}.5": "UND"},
}

############################################################################################################

passed_comprehension = data_gold.reset_index(drop=True)  # keep names consistent to trivago experiment (no comprehension check here)

############################################################################################################

# Extract the category for each annotation

NUM_ANNOTATORS = passed_comprehension.shape[0]
NUM_ITEMS = 75  # fixed number of items
START_BLOCK = 5

# Reseting the index
passed_comprehension = passed_comprehension.reset_index(drop=True)

# Function to extract category for a specific annotation
def extract_category_from_annotation(df, item, annotator):

    global per_step, categories_reverse

    for category, questions in per_step.items():
        matching = all(df.loc[annotator, question.format(item=item)] == value for question, value in questions.items())
        if matching:
            return categories_reverse[category]
    
    return None

# Create a DataFrame to store the results
results = pd.DataFrame(index=range(NUM_ANNOTATORS))

# Loop through each item and each annotator, and store the results in the DataFrame
for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK):
    item_results = []
    for annotator in range(NUM_ANNOTATORS):
        category = extract_category_from_annotation(passed_comprehension, item, annotator)
        if category is not None:
            item_results.append(categories[category])
        else:
            item_results.append(np.nan)
    results[f"Item {item - (START_BLOCK - 1)}"] = item_results

# Calculate Krippendorff's alpha
print("Krippendorff's alpha for overall category:", krippendorff.alpha(reliability_data=results, level_of_measurement="nominal"))

# count unique categories and plot (overall unique category counts)
unique_category_counts = results.apply(lambda row: row.value_counts(), axis=1).fillna(0).astype(int).rename(columns=categories_reverse, index=lambda x: f"Annotator {x}").sum().rename("Count")
fig = px.bar(unique_category_counts, y="Count")
fig = fig.update_layout(title="Category distribution", xaxis_title="Category", yaxis_title="Count")
fig.show()

# plot of category, item and annotator
results_filtered = (
    results
    .reset_index()
    .melt(id_vars='index', var_name='Item', value_name='Category')
    .dropna(subset=['Category'])
    .assign(Annotator=lambda x: 'Annotator ' + (x['index'] + 1).astype(str))
    .drop(columns='index')
    .apply(lambda x: x.map(categories_reverse) if x.name == 'Category' else x)
)
fig = px.scatter(results_filtered, x="Item", y="Annotator", color="Category",
                 title="Category distribution per item and annotator",
                 category_orders={"Category": list(categories_reverse.values()),
                                  "Item": list(results.columns),
                                  "Annotator": list(results_filtered['Annotator'].unique())},
                 color_discrete_map=categories_reverse)

fig.show()

############################################################################################################

# Calculate IAA for ambiguity

updated_names = {"There is one or more connective precedence ambiguity.": "Connective precedence", "There is one or more negation scope ambiguity.": "Negation scope", "There is one or more quantifier scope ambiguity.": "Quantifier scope"}

# Retrieve the .3 and .4 fields from the annotation considering NUM_ITEMS and START_BLOCK
ambiguity_answer = passed_comprehension[[f"Q{item}.3" for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK)]].replace({"Yes": 1, "No": 0})
print("Krippendorff's alpha for ambiguity:", krippendorff.alpha(reliability_data=ambiguity_answer, level_of_measurement="nominal"))

# ambiguities motivations
ambiguity_motivation = passed_comprehension[[f"Q{item}.4" for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK)]]
ambiguity_motivation.columns = results.columns  # rename columns to match the results
ambiguity_motivation = ambiguity_motivation.dropna(axis=1, how='all')
ambiguity_motivation = ambiguity_motivation.applymap(lambda x: x.split(",") if pd.notna(x) else np.nan)
ambiguity_motivation = ambiguity_motivation.applymap(lambda x: [updated_names.get(i, i) for i in x] if isinstance(x, list) else x)

# count motivations per item
ambiguity_motivation_counts = ambiguity_motivation.apply(lambda x: x.explode().value_counts()).fillna(0).astype(int).rename(columns=updated_names)
most_ambiguous_items = ambiguity_motivation_counts.sum(axis=0).sort_values(ascending=False).head(2)
print("Most ambiguous items:")
print(most_ambiguous_items)
print("Ambiguity motivation distribution:")
print(ambiguity_motivation_counts.sum(axis=1).sort_values(ascending=False))

# ############################################################################################################

# count I |= O
i_o_answer = results.apply(lambda row: row.apply(lambda x: 1 if x in [1, 3] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 3: I -> O
print("Krippendorff's alpha for I -> O:", krippendorff.alpha(reliability_data=i_o_answer, level_of_measurement="nominal"))

# count O |= I
o_i_answer = results.apply(lambda row: row.apply(lambda x: 1 if x in [1, 5] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 5: O -> I
print("Krippendorff's alpha for O -> I:", krippendorff.alpha(reliability_data=o_i_answer, level_of_measurement="nominal"))

# ############################################################################################################

# General questions

# complexity of the task
print("Complexity of the task (mean):", passed_comprehension["Q80.4_1"].mean())

# reason for the complexity:

# training helped?
print("Helpfulness of the training (mean):", passed_comprehension["Q80.6_1"].mean())

# clarity of the guidelines
print("Clarity of the guidelines (mean):", passed_comprehension["Q80.7_1"].mean())

# easiness of the UI
print("Easiness of the UI (mean):", passed_comprehension["Q80.10_1"].mean())