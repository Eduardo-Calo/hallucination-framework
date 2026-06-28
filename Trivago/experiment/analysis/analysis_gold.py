import krippendorff
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.subplots as ps
import re
from scipy.spatial.distance import hamming
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_score
import json
from collatex import *  # http://interedition.github.io/collatex/pythonport.html
import spacy


# Load the data
data = pd.read_csv("experiment/gold_label/albert_edu_kees_gold.csv")

# Zipping first two rows
questions_mapping = dict(zip(data.columns, data.iloc[0]))

# Filter who didn't finish the experiment: drop rows with "Finished" == False (Prolific time out)
data = data[data["Finished"] == "True"]

# Keep questions columns
data = data[[col for col in data.columns if col.startswith("Q")]].iloc[:, 2:]

# Cast to numeric
def cast_str_to_int(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            contains_number = df[column].apply(lambda x: bool(re.search(r'^\d+$', str(x))))
            if contains_number.all():
                df[column] = df[column].astype(int)
    return df

data = cast_str_to_int(data)

# map categories to numbers
categories = {
    "0a: Well-matched": 1,
    "0b: Well-matched with harmless information": 2,
    "1a: Output too weak": 3,
    "1b: Output too weak with harmless information": 4,
    "2a: Output too strong": 5,
    "2b: Output contradictory": 6,
    "3a: Input and Output independent": 7,
    "3b: Input and Output contradictory": 8,
}

# reverse mapping
categories_reverse = {v: k for k, v in categories.items()}

# map categories to per step answers: e.g., (6, 9, 10, if YES: O -> I) 
per_step = {
    1: {"Q{item}.3": "No", "Q{item}.5": "Yes", "Q{item}.6": "Yes"},
    2: {"Q{item}.3": "No", "Q{item}.5": "No", "Q{item}.7": "No", "Q{item}.9": "Yes"},
    3: {"Q{item}.3": "No", "Q{item}.5": "Yes", "Q{item}.6": "No"},
    4: {"Q{item}.3": "No", "Q{item}.5": "No", "Q{item}.7": "No", "Q{item}.9": "No"},
    5: {"Q{item}.3": "No", "Q{item}.5": "No", "Q{item}.7": "Yes", "Q{item}.10": "Yes"},
    6: {"Q{item}.3": "Yes"},
    7: {"Q{item}.3": "No", "Q{item}.5": "No", "Q{item}.7": "Yes", "Q{item}.10": "No", "Q{item}.11": "No"},
    8: {"Q{item}.3": "No", "Q{item}.5": "No", "Q{item}.7": "Yes", "Q{item}.10": "No", "Q{item}.11": "Yes"}
}

############################################################################################################

passed_comprehension = data[data['Q30.2'].notna()]
failed_comprehension = data[data['Q30.2'].isna()]

# length of the data
length_data = passed_comprehension.shape[0] + failed_comprehension.shape[0]

print(f"Passed comprehension: {passed_comprehension.shape[0]} ({passed_comprehension.shape[0] / length_data * 100:.2f}%)")
print(f"Failed comprehension: {failed_comprehension.shape[0]} ({failed_comprehension.shape[0] / length_data * 100:.2f}%)")

############################################################################################################

# Extract the category for each annotation

NUM_ANNOTATORS = passed_comprehension.shape[0]
NUM_ITEMS = 25  # fixed number of items
START_BLOCK = 5  # start from 5 because the first 3 blocks are not questions and 4 is the comprehension check

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

# macro category agreement

# map categories to macro categories
macro_categories = {
    1: 1,  # 0a
    2: 1,  # 0b
    3: 2,  # 1a
    4: 2,  # 1b
    5: 3,  # 2a
    6: 4,  # 2b
    7: 5,  # 3a
    8: 6,  # 3b
}

macro_categories_reverse = {1: "Well-matched", 2: "Output too weak", 3: "Output too strong", 4: "Output contradictory", 5: "Input and Output independent", 6: "Input and Output contradictory"}

# map results to macro categories
results_macro = results.applymap(lambda x: macro_categories.get(x, np.nan))

# Calculate Krippendorff's alpha
print("Krippendorff's alpha for macro category:", krippendorff.alpha(reliability_data=results_macro, level_of_measurement="nominal"))

# plot of macro category, item and annotator
results_filtered_macro = (
    results_macro
    .reset_index()
    .melt(id_vars='index', var_name='Item', value_name='Category')
    .dropna(subset=['Category'])
    .assign(Annotator=lambda x: 'Annotator ' + (x['index'] + 1).astype(str))
    .drop(columns='index')
    .apply(lambda x: x.map(macro_categories_reverse) if x.name == 'Category' else x)
)
fig = px.scatter(results_filtered_macro, x="Item", y="Annotator", color="Category",
                 title="Macro category distribution per item and annotator",
                 category_orders={"Category": list(macro_categories.values()),
                                  "Item": list(results.columns),
                                  "Annotator": list(results_filtered['Annotator'].unique())},
                 color_discrete_map=macro_categories_reverse)

fig.show()

############################################################################################################

# count I |= O
i_o_answer = results.apply(lambda row: row.apply(lambda x: 1 if x in [1, 3] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 3: I -> O
print("Krippendorff's alpha for I -> O:", krippendorff.alpha(reliability_data=i_o_answer, level_of_measurement="nominal"))

# count O |= I
o_i_answer = results.apply(lambda row: row.apply(lambda x: 1 if x in [1, 2, 5] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1, 2 and 5: O -> I
print("Krippendorff's alpha for O -> I:", krippendorff.alpha(reliability_data=o_i_answer, level_of_measurement="nominal"))

# count factually wrong
factually_wrong_answer = results.apply(lambda row: row.apply(lambda x: 1 if x in [5, 6, 7] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 5, 6, 7: factually wrong information
print("Krippendorff's alpha for factually wrong information:", krippendorff.alpha(reliability_data=factually_wrong_answer, level_of_measurement="nominal"))

############################################################################################################

# IAA with final slider value

slider_values = pd.DataFrame(index=range(NUM_ANNOTATORS))

# Loop through each item and each annotator, and store the results in the DataFrame
for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK):
    item_results = []
    for annotator in range(NUM_ANNOTATORS):
        item_results.append(passed_comprehension.loc[annotator, f"Q{item}.13_1"])
    slider_values[f"Item {item - (START_BLOCK - 1)}"] = item_results

slider_values = slider_values.astype(float)

# Calculate Krippendorff's alpha
print("Krippendorff's alpha for slider values:", krippendorff.alpha(reliability_data=slider_values, level_of_measurement="ordinal"))  # unexpectedly lower than the category agreement
# Try discritizing the slider values and see if it improves the agreement
slider_values_binned = slider_values.apply(lambda x: pd.cut(x, bins=[1, 3, 5, 8], labels=[1, 2, 3], right=False)).astype(float)
print("Krippendorff's alpha for binned slider values:", krippendorff.alpha(reliability_data=slider_values_binned, level_of_measurement="ordinal"))  # even lower

# plot the slider values per item and per annotator
slider_values_counts = (
    slider_values
    .reset_index()
    .melt(id_vars='index', var_name='Item', value_name='Slider value')
    .dropna(subset=['Slider value'])
    .assign(Annotator=lambda x: 'Annotator ' + (x['index'] + 1).astype(str))
    .drop(columns='index')
    .apply(lambda x: x.map(int).map(str) if x.name == 'Slider value' else x)
)

fig = px.scatter(slider_values_counts, x="Item", y="Annotator", color="Slider value",
                    title="Slider values distribution per item and annotator",
                    category_orders={"Slider value": sorted(slider_values_counts["Slider value"].unique()),
                                     "Item": list(results.columns),
                                     "Annotator": list(results_filtered['Annotator'].unique())},
                    color_discrete_map=dict(zip(sorted(slider_values_counts["Slider value"].unique()), px.colors.sequential.Blues))
                    )

fig.show()

############################################################################################################

# load Input and Output
raw_data = pd.read_csv("experiment/data_processed.tsv", sep="\t").dropna(subset=["block_Q"]).reset_index(drop=True)
raw_data["item"] = raw_data["block_Q"].apply(lambda x: f"Item {int(x) - 4}")

# split Input and Output into dict with index : value
raw_data["input"] = raw_data["input"].apply(lambda x: {i: k for i, k in enumerate(x.split(), 1)})
raw_data["output"] = raw_data["output"].apply(lambda x: {i: k for i, k in enumerate(x.split(), 1)})

# separate Input and Output
raw_data_input = raw_data.set_index("item")["input"]
raw_data_output = raw_data.set_index("item")["output"]

############################################################################################################

# Count divergent information per item and per annotator

# extract divergent information if available
def extract_divergent_information_input(df, item, annotator):
    if f"Q{item}.1_1" in df.columns:
        if pd.notna(df.loc[annotator, f"Q{item}.3"]):  # if the control cell contains something, aka the annotator performed that annotation in the latin square
            info = df.loc[annotator, f"Q{item}.1_1"]
            if pd.notna(info):
                return info
            return "None"  # if they didn't perform the annotation, return None, to be consistent when computing results
    return np.nan  # return nan to those who didn't perform that annotation

def extract_divergent_information_output(df, item, annotator):
    if f"Q{item}.2_1" in df.columns:
        if pd.notna(df.loc[annotator, f"Q{item}.3"]):
            info = df.loc[annotator, f"Q{item}.2_1"]
            if pd.notna(info):
                return info
            return "None"
    return np.nan

# Remove who hasn't highlighted divergent information
passed_comprehension.drop(passed_comprehension.tail(1).index, inplace=True)
NUM_ANNOTATORS = passed_comprehension.shape[0]

# Create a DataFrame to store the results
results_divergent_input = pd.DataFrame(index=range(NUM_ANNOTATORS))
results_divergent_output = pd.DataFrame(index=range(NUM_ANNOTATORS))

# Loop through each item and each annotator, and store the results in the DataFrame
for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK):  # start from 5 because the first 3 blocks are not questions and 4 is the comprehension check
    item_results_input = []
    item_results_output = []
    for annotator in range(NUM_ANNOTATORS):
        item_results_input.append(extract_divergent_information_input(passed_comprehension, item, annotator))
        item_results_output.append(extract_divergent_information_output(passed_comprehension, item, annotator))
    results_divergent_input[f"Item {item - (START_BLOCK - 1)}"] = item_results_input
    results_divergent_output[f"Item {item - (START_BLOCK - 1)}"] = item_results_output

# process divergent information
def process_divergent(input_str):
    # Split the input string by comma to separate key-value pairs
    pairs = input_str.split(',')

    # If empty string, add a comma to the previous element in the list and remove the empty string
    for i in range(1, len(pairs)):
        if pairs[i] == '':
            pairs[i - 1] += ','
    pairs = [pair for pair in pairs if pair]

    # Split each pair by colon to separate key and value and store them in a dictionary
    input_dict = {int(pair.split(':')[0]): pair.split(':')[1].strip() for pair in pairs}
    
    return input_dict

results_divergent_input_proc = results_divergent_input.applymap(lambda x: process_divergent(x) if isinstance(x, str) and x != "None" else (np.nan if pd.isna(x) else {}))
results_divergent_output_proc = results_divergent_output.applymap(lambda x: process_divergent(x) if isinstance(x, str) and x != "None" else (np.nan if pd.isna(x) else {}))

# count divergent information per item and per annotator
divergent_info_counts_input = results_divergent_input_proc.apply(lambda row: row.apply(lambda x: len(x) if isinstance(x, dict) else np.nan), axis=1)
divergent_info_counts_output = results_divergent_output_proc.apply(lambda row: row.apply(lambda x: len(x) if isinstance(x, dict) else np.nan), axis=1)

# plot divergent information counts per item and per annotator (normalized by the number of tokens in the input and output)
divergent_info_counts_input_counts = (
    divergent_info_counts_input
    .reset_index()
    .melt(id_vars='index', var_name='Item', value_name='Count')
    .dropna(subset=['Count'])
    .assign(Annotator=lambda x: 'Annotator ' + (x['index'] + 1).astype(str))
    .drop(columns='index')
    .apply(lambda x: x.map(int) if x.name == 'Count' else x)
)
divergent_info_counts_input_counts["Ratio"] = divergent_info_counts_input_counts["Count"] / divergent_info_counts_input_counts["Item"].map(raw_data_input.apply(len))
fig1 = px.scatter(divergent_info_counts_input_counts, x="Item", y="Annotator", color="Ratio",
                 title="Divergent information counts per item and annotator (Input)",
                 category_orders={"Item": list(results.columns),
                                  "Annotator": list(results_filtered['Annotator'].unique())},
                )

divergent_info_counts_output_counts = (
    divergent_info_counts_output
    .reset_index()
    .melt(id_vars='index', var_name='Item', value_name='Count')
    .dropna(subset=['Count'])
    .assign(Annotator=lambda x: 'Annotator ' + (x['index'] + 1).astype(str))
    .drop(columns='index')
    .apply(lambda x: x.map(int) if x.name == 'Count' else x)
)
divergent_info_counts_output_counts["Ratio"] = divergent_info_counts_output_counts["Count"] / divergent_info_counts_output_counts["Item"].map(raw_data_output.apply(len))
fig2 = px.scatter(divergent_info_counts_output_counts, x="Item", y="Annotator", color="Ratio",
                 title="Divergent information counts per item and annotator (Output)",
                 category_orders={"Item": list(results.columns),
                                  "Annotator": list(results_filtered['Annotator'].unique())},
                )

fig = ps.make_subplots(
    rows=1, cols=2,
    subplot_titles=("Input", "Output"),
    shared_yaxes=True,
    x_title="Item"
)

fig.add_traces(fig1.data, rows=1, cols=1)
fig.add_traces(fig2.data, rows=1, cols=2)
fig.update_layout(title="Ratio of divergent information annotated per item and annotator", yaxis_title="Annotator", coloraxis_colorbar=dict(title="Ratio"))
fig.update_yaxes(autorange="reversed")

fig.show()

#############################################################################################################

# Mismatch highlights - answers (unreliable annotators)

# annotators that highlighted divergent info in O should answer "No" to "I |= O" (regarless of factually wrong information; that comes later)
# annotators that didn't highlight divergent info in O should answer "Yes" to "I |= O"
mismatch_io_df = i_o_answer.drop(i_o_answer.tail(1).index)
mismatch_io_df[(divergent_info_counts_output > 0) & (i_o_answer == 0)] = -1
mismatch_io_df[(divergent_info_counts_output == 0) & (i_o_answer == 1)] = -1
# count good annotations per annotator (out of 5 items)
count_good_io = mismatch_io_df.apply(lambda x: x.value_counts().get(-1, 0), axis=1)

# annotators that highlighted divergent info in I should answer "No" to "O |= I"
# annotators that didn't highlight divergent info in I should answer "Yes" to "O |= I"
mismatch_oi_df = o_i_answer.drop(o_i_answer.tail(1).index)
mismatch_oi_df[(divergent_info_counts_input > 0) & (o_i_answer == 0)] = -1
mismatch_oi_df[(divergent_info_counts_input == 0) & (o_i_answer == 1)] = -1
# count good annotations per annotator (out of 5 items)
count_good_oi = mismatch_oi_df.apply(lambda x: x.value_counts().get(-1, 0), axis=1)

# pinpoint annotators that are unreliable (good count < 50 I - O (all))
unreliable_annotators = count_good_io + count_good_oi
print("Unreliable annotators (mismatches out of 50):", dict(zip([f"Annotator {i+1}" for i in unreliable_annotators[unreliable_annotators < 50].index], 50 - unreliable_annotators[unreliable_annotators < 50])))

#############################################################################################################

# Remove those who haven't highlighted divergent information
slider_values = slider_values.drop(slider_values.tail(1).index)

# Correlation (average slider value per average ratio of divergent information)
divergent_info_ratio_input = divergent_info_counts_input.copy()

for col in divergent_info_ratio_input.columns:
    divergent_info_ratio_input[col] = divergent_info_ratio_input[col] / len(raw_data_input[col])
divergent_info_ratio_input = divergent_info_ratio_input.where(divergent_info_counts_input.notnull(), np.nan)

divergent_info_ratio_output = divergent_info_counts_output.copy()

for col in divergent_info_ratio_output.columns:
    divergent_info_ratio_output[col] = divergent_info_ratio_output[col] / len(raw_data_output[col])
divergent_info_ratio_output = divergent_info_ratio_output.where(divergent_info_counts_output.notnull(), np.nan)

divergent_info_ratio_input_mean = divergent_info_ratio_input.mean(axis=0)
divergent_info_ratio_output_mean = divergent_info_ratio_output.mean(axis=0)

correlation_df = pd.DataFrame({"Divergent information ratio (Input)": divergent_info_ratio_input_mean, "Divergent information ratio (Output)": divergent_info_ratio_output_mean, "Slider value": slider_values.mean(axis=0)})

#############################################################################################################

# Factually wrong information over divergent information (count and ratio)

# extract factually wrong information if available
def extract_factually_wrong_information(df, item, annotator):
    if f"Q{item}.8" in df.columns:
        if pd.notna(df.loc[annotator, f"Q{item}.3"]):  # if the control cell contains something, aka the annotator performed that annotation in the latin square
            info = df.loc[annotator, f"Q{item}.8"]
            if pd.notna(info):
                return info
            return "None"  # if they didn't perform the annotation, return None, to be consistent when computing results
    return np.nan  # return nan to those who didn't perform that annotation

# Create a DataFrame to store the results
results_factually_wrong = pd.DataFrame(index=range(NUM_ANNOTATORS))

# Loop through each item and each annotator, and store the results in the DataFrame
for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK):  # start from 5 because the first 3 blocks are not questions and 4 is the comprehension check
    item_results = []
    for annotator in range(NUM_ANNOTATORS):
        item_results.append(extract_factually_wrong_information(passed_comprehension, item, annotator))
    results_factually_wrong[f"Item {item - (START_BLOCK - 1)}"] = item_results


# Process factually wrong information
# count = number tokens of factually wrong information / number tokens of divergent information
factually_wrong_info_counts = results_factually_wrong.apply(lambda row: row.apply(lambda x: 0 if x == "None" else len(x.split(' ')) if isinstance(x, str) else np.nan), axis=1)
# handle 0/0 division
mask = (factually_wrong_info_counts == 0) & (divergent_info_counts_output == 0)
factually_wrong_info_ratio_output = factually_wrong_info_counts.div(divergent_info_counts_output, fill_value=0).where(~mask,0)
# replace inf with 0
factually_wrong_info_ratio_output = factually_wrong_info_ratio_output.replace(np.inf, 0)


# Plot factually wrong information ratio per item and per annotator over divergent information
factually_wrong_info_ratio = (
    factually_wrong_info_ratio_output
    .reset_index()
    .melt(id_vars='index', var_name='Item', value_name='Ratio')
    .dropna(subset=['Ratio'])
    .assign(Annotator=lambda x: 'Annotator ' + (x['index'] + 1).astype(str))
    .drop(columns='index')
    .assign(Ratio=lambda x: x['Ratio'].apply(lambda x: 1 if x > 1 else x))
)

fig = px.scatter(factually_wrong_info_ratio, x="Item", y="Annotator", color="Ratio",
                    title="Factually wrong information ratio over divergent information per item and annotator",
                    category_orders={"Item": list(results.columns),
                                    "Annotator": list(results_filtered['Annotator'].unique())},
                    color_continuous_scale="Reds")

fig.show()

# Compute the correlation between slider value and factually wrong information ratios
factually_wrong_info_ratio_overall = factually_wrong_info_counts.copy()

for col in factually_wrong_info_ratio_overall.columns:
    factually_wrong_info_ratio_overall[col] = factually_wrong_info_ratio_overall[col] / len(raw_data_output[col])
factually_wrong_info_ratio_overall = factually_wrong_info_ratio_overall.where(factually_wrong_info_counts.notnull(), np.nan)

factually_wrong_info_ratio_mean = factually_wrong_info_ratio_overall.mean(axis=0).fillna(0)
corr_df = pd.DataFrame({"Factually wrong information ratio": factually_wrong_info_ratio_mean, "Slider value": slider_values.mean(axis=0)})

#############################################################################################################

# Plot the correlation (average slider value per average ratio of divergent information and factually wrong information)
correlation_df_combined = pd.concat([correlation_df, corr_df], axis=1)
correlation_df_combined = correlation_df_combined.loc[:,~correlation_df_combined.columns.duplicated()]

# Replace 0 with a very small positive number for visualization purposes
correlation_df_combined = correlation_df_combined.replace(0, 1e-10)

# Create the figure
fig = px.scatter(correlation_df_combined, x="Slider value", y=["Divergent information ratio (Input)", "Divergent information ratio (Output)", "Factually wrong information ratio"], trendline="ols", log_y=True, trendline_options=dict(log_y=True))

# Update layout
fig.update_layout(
    title="Correlation between slider value and divergent/factually wrong information ratio",
    xaxis_title="Slider value",
    yaxis_title="Ratio",
    legend_title="",
    yaxis=dict(tickvals=[1e-10, 0.001, 0.01, 0.1, 0.5, 1], ticktext=["0", "0.001", "0.01", "0.1", "0.5", "1"])
)

fig.show()

# Make a table with the correlation values
print("Correlation between slider value and divergent/factually wrong information:")
correlation_table = pd.DataFrame({
    "Divergent information ratio (Input)": pearsonr(correlation_df["Divergent information ratio (Input)"], correlation_df["Slider value"]),  # significant negative correlation
    "Divergent information ratio (Output)": pearsonr(correlation_df["Divergent information ratio (Output)"], correlation_df["Slider value"]),  # non-significant correlation (may be because of fluff; compare with factually wrong information)
    "Factually wrong information ratio": pearsonr(corr_df["Factually wrong information ratio"], corr_df["Slider value"])  # significant negative correlation
}, index=["r", "p"])

print(correlation_table.T)

#############################################################################################################

# Compute agreement on vectors of occurences (see WHAT people consider as divergent)
# AND
# Vectors of occurences for factually wrong information (to see what people consider as factually wrong vs. fluff)

# INPUT: str that doesn't align with dict and may have typos
# OUTPUT: factually_wrong_dicts = key: value (same as dict_tokens of the divergent info)

nlp = spacy.load('en_core_web_sm')

# get the occurrence vector
def get_occurrence_vector(dict1, *dicts):
    # Get counts for each token
    vector_counts = [sum(key in d for d in dicts) for key in range(1, max(dict1, default=0) + 1)]
    
    return vector_counts

# add invisible whitespace to tokens to make them unique: helper function for visualization of labels
def add_incremental_whitespace(tokens):
    token_counts = {}
    modified_tokens = []
    for token in tokens:
        if token in token_counts:
            token_counts[token] += 1
        else:
            token_counts[token] = 1
        modified_token = token + "\u200b" * (token_counts[token] - 1)
        modified_tokens.append(modified_token)
    return modified_tokens

# compute the average hamming similarity: helper function for computing the average similarity
def average_hamming_similarity(vectors_list):
    similarities = []
    for i in range(len(vectors_list)):
        for j in range(i+1, len(vectors_list)):
            similarity = 1 - hamming(vectors_list[i], vectors_list[j])  # 1 - hamming distance
            similarities.append(similarity)

    return np.mean(similarities)

# helper function for computing Krippendorff's alpha (handle perfect agreement)
def compute_krippendorff_alpha(data):
    unique_values = set()
    for row in data:
        unique_values.update(row)

    if len(unique_values) <= 1:
        return 1.0  # Perfect agreement implied by lack of variability
    else:
        # Calculate Krippendorff's alpha
        return krippendorff.alpha(data, level_of_measurement="nominal")
    
# compute average jaccard similarity
def average_jaccard_similarity(vectors_list):
    similarities = []
    for i in range(len(vectors_list)):
        for j in range(i+1, len(vectors_list)):
            similarity = jaccard_score(vectors_list[i], vectors_list[j], zero_division=1)
            similarities.append(similarity)

    return np.mean(similarities)

def process_tokens(dict_tokens):    
    # Initialize a list to store (original span, tokens, lemmas, key) tuples
    dict_tokens_lemmatized = []
    
    # Process each value in the dictionary separately
    for key, value in dict_tokens.items():
        # Process the value with spaCy
        doc = nlp(value)
        # Get the tokens of the doc
        tokens = [str(key) + "[SEP]" + token.text for token in doc]
        # Get the lemmas of the doc
        lemmas = [token.lemma_ for token in doc]
        # Append the original span, its tokens, lemmas, and its key to the list
        dict_tokens_lemmatized.append((value, tokens, lemmas, key))
    
    flatten_tokens = [token for _, tokens, _, _ in dict_tokens_lemmatized for token in tokens]
    flatten_lemmas = [lemma for _, _, lemmas, _ in dict_tokens_lemmatized for lemma in lemmas]
    
    assert len(flatten_tokens) == len(flatten_lemmas)
    
    return list(zip(flatten_tokens, flatten_lemmas))

def prepare_json(text_a, text_b):
    result = {"witnesses": []}

    # Create witness for text_a (dict_tokens)
    text_a_witness = {"id": "A", "tokens": []}
    for tup in text_a:
        token, lemma = tup
        text_a_witness["tokens"].append({"t": token, "n": lemma})
    result["witnesses"].append(text_a_witness)

    # Create witness for text_b (text)
    text_b_witness = {"id": "B", "tokens": []}
    for tup in text_b:
        token, lemma = tup
        text_b_witness["tokens"].append({"t": token, "n": lemma})
    result["witnesses"].append(text_b_witness)
    
    return json.dumps(result, indent=4)

def align_tokens(dict_tokens, text):
    # Process the tokens in the dictionary
    dict_tokens_processed = process_tokens(dict_tokens)
    
    # Process the lemmatized text
    text_lemmatized = [(token.text, token.lemma_) for token in nlp(text.replace("\n", " "))]
    
    # Tokenize the text and the dictionary tokens
    json_result = prepare_json(dict_tokens_processed, text_lemmatized)
    
    # Align the tokens
    collation = Collation()
    alignment_table = collate(json.loads(json_result), near_match=True, segmentation=False, layout='vertical')
    # print(alignment_table)
    
    # Get the original tokens and the alignment
    orig_tokens = alignment_table.rows[0].to_list_of_strings()
    alignment = alignment_table.rows[1].to_list_of_strings()
    
    return orig_tokens, alignment

def construct_factually_wrong_dict(orig_tokens, alignment):
    # if match keep the element in list 1
    # split by SEP
    # concatenate consecutive elements in list 1 that have the same key
    # create dict with key and value
    
    factually_wrong_dict = {}
    
    i = 0
    while i < len(orig_tokens):
        orig_token = orig_tokens[i]
        align = alignment[i]

        if align is None or orig_token is None:
            i += 1
            continue

        orig_token_parts = orig_token.split("[SEP]")

        # Check if the next token starts with the same key
        key = int(orig_token_parts[0])
        value = orig_token_parts[1]

        j = i + 1
        while j < len(orig_tokens):
            
            # Check if the alignment or the original token is None
            if orig_tokens[j] is None or alignment[j] is None:
                break

            next_orig_token = orig_tokens[j].split("[SEP]")
            if next_orig_token[0] != orig_token_parts[0]:
                break

            # Append the value of the next token
            value += next_orig_token[1]
            j += 1

        factually_wrong_dict[key] = value
        i = j  # Move to the next token after the last one included in this key's value
    
    return factually_wrong_dict

def compute_item_similarity(item_num, show=False):

    global raw_data_input, raw_data_output, results_divergent_input_proc, results_divergent_output_proc, results_factually_wrong

    # Get the input and output dictionaries for the current item
    input_dict = raw_data_input[f"Item {item_num}"]
    output_dict = raw_data_output[f"Item {item_num}"]

    # Get the factually wrong information strings for the current item
    factually_wrong_strs = results_factually_wrong[f"Item {item_num}"].dropna()

    # Get the divergent and factually wrong information dictionaries for the current item
    divergent_input_dicts = results_divergent_input_proc[f"Item {item_num}"].dropna()
    divergent_output_dicts = results_divergent_output_proc[f"Item {item_num}"].dropna()
    factually_wrong_dicts = factually_wrong_strs.apply(lambda x: construct_factually_wrong_dict(*align_tokens(output_dict, x)) if x != "None" else {})

    # Get the vectors of occurrences for the input and output and the factually wrong information
    input_vectors = divergent_input_dicts.apply(lambda x: get_occurrence_vector(input_dict, x))
    output_vectors = divergent_output_dicts.apply(lambda x: get_occurrence_vector(output_dict, x))
    factually_wrong_vectors = factually_wrong_dicts.apply(lambda x: get_occurrence_vector(output_dict, x))

    # To df
    input_vectors_df = pd.DataFrame(input_vectors.tolist(), index=list(input_vectors.index), columns=list(input_dict.values()))
    output_vectors_df = pd.DataFrame(output_vectors.tolist(), index=list(output_vectors.index), columns=list(output_dict.values()))
    factually_wrong_vectors_df = pd.DataFrame(factually_wrong_vectors.tolist(), index=list(factually_wrong_vectors.index), columns=list(output_dict.values()))

    # Compute IAA (outdated)
    iaa_input = compute_krippendorff_alpha(input_vectors_df.values)
    iaa_output = compute_krippendorff_alpha(output_vectors_df.values)
    iaa_factually_wrong = compute_krippendorff_alpha(factually_wrong_vectors_df.values)

    # Compute the average Hamming similarity (outdated)
    average_similarity_input = average_hamming_similarity(input_vectors.tolist())
    average_similarity_output = average_hamming_similarity(output_vectors.tolist())
    average_similarity_factually_wrong = average_hamming_similarity(factually_wrong_vectors.tolist())

    # Compute Jaccard similarity
    jaccard_similarity_input = average_jaccard_similarity(input_vectors.tolist())
    jaccard_similarity_output = average_jaccard_similarity(output_vectors.tolist())
    jaccard_similarity_factually_wrong = average_jaccard_similarity(factually_wrong_vectors.tolist())

    # Get the summed vectors
    input_vector_final = get_occurrence_vector(input_dict, *divergent_input_dicts)
    output_vector_final = get_occurrence_vector(output_dict, *divergent_output_dicts)
    factually_wrong_vector_final = get_occurrence_vector(output_dict, *factually_wrong_dicts)

    if show:
        # Visualize the final vector: heatmap of the occurrence of each token
        fig = ps.make_subplots(
            rows=3, cols=1,
            subplot_titles=("Divergent information (Input)", "Divergent information (Output)", "Factually wrong information"),
            shared_xaxes=False,
            x_title="",
            y_title="",
            vertical_spacing=0.2
        )

        # Add each heatmap
        fig1 = px.imshow([input_vector_final], x=add_incremental_whitespace(list(input_dict.values())), labels=dict(x="Token", color="Occurrence", y=""), color_continuous_scale="Reds", title="", text_auto=True)
        fig2 = px.imshow([output_vector_final], x=add_incremental_whitespace(list(output_dict.values())), labels=dict(x="Token", color="Occurrence", y=""), color_continuous_scale="Reds", title="", text_auto=True)
        fig3 = px.imshow([factually_wrong_vector_final], x=add_incremental_whitespace(list(output_dict.values())), labels=dict(x="Token", color="Occurrence", y=""), color_continuous_scale="Reds", title="", text_auto=True)

        # Add traces to subplots
        fig.add_traces(fig1.data, rows=1, cols=1)
        fig.add_traces(fig2.data, rows=2, cols=1)
        fig.add_traces(fig3.data, rows=3, cols=1)

        # Update layout
        fig.update_layout(title=f"Vectors of occurrences for Item {item_num}")
        # update colorbar
        fig.update_coloraxes(colorscale="Reds")
        # Hide y-axis tick labels
        fig.update_yaxes(showticklabels=False)

        fig.show()

    return average_similarity_input, average_similarity_output, average_similarity_factually_wrong, iaa_input, iaa_output, iaa_factually_wrong, jaccard_similarity_input, jaccard_similarity_output, jaccard_similarity_factually_wrong

# compute_item_similarity(13, show=True)

similarities_input = []
similarities_output = []
similarities_factually_wrong = []
iaas_input = []
iaas_output = []
iaas_factually_wrong = []
jaccards_input = []
jaccards_output = []
jaccards_factually_wrong = []

for item in range(1, NUM_ITEMS + 1):
    similarity_input, similarity_output, similarity_factually_wrong, iaa_input, iaa_output, iaa_factually_wrong, jaccard_input, jaccard_output, jaccard_factually_wrong = compute_item_similarity(item, show=False)
    similarities_input.append(similarity_input)
    similarities_output.append(similarity_output)
    similarities_factually_wrong.append(similarity_factually_wrong)
    iaas_input.append(iaa_input)
    iaas_output.append(iaa_output)
    iaas_factually_wrong.append(iaa_factually_wrong)
    jaccards_input.append(jaccard_input)
    jaccards_output.append(jaccard_output)
    jaccards_factually_wrong.append(jaccard_factually_wrong)

#############################################################################################################

# Agreement means Jaccard on divergent information in input and output and factually wrong information

jaccard_df = pd.DataFrame({"Divergent information (Input)": jaccards_input, "Divergent information (Output)": jaccards_output, "Factually wrong information": jaccards_factually_wrong}, index="Item " + pd.Series(range(1, NUM_ITEMS + 1)).astype(str))

fig = ps.make_subplots(
    rows=3, cols=1,
    subplot_titles=("Divergent information (Input)", "Divergent information (Output)", "Factually wrong information"),
    shared_xaxes=True,
    x_title="Item"
)

# Add each heatmap
fig1 = px.imshow([jaccards_input],
                    labels=dict(x="Item", color="Average Jaccard similarity", y="Divergent information in Input"),
                    x=[f"Item {i}" for i in range(1, NUM_ITEMS + 1)],
                    color_continuous_scale="Reds", title="",
                    text_auto=True
                    )
fig2 = px.imshow([jaccards_output],
                    labels=dict(x="Item", color="Average Jaccard similarity", y="Divergent information in Output"),
                    x=[f"Item {i}" for i in range(1, NUM_ITEMS + 1)],
                    color_continuous_scale="Reds", title="",
                    text_auto=True
                    )
fig3 = px.imshow([jaccards_factually_wrong],
                    labels=dict(x="Item", color="Average Jaccard similarity", y="Factually wrong information"),
                    x=[f"Item {i}" for i in range(1, NUM_ITEMS + 1)],
                    color_continuous_scale="Reds", title="",
                    text_auto=True
                    )

# Add traces to subplots
fig.add_traces(fig1.data, rows=1, cols=1)
fig.add_traces(fig2.data, rows=2, cols=1)
fig.add_traces(fig3.data, rows=3, cols=1)

# Update layout
fig.update_layout(title="Agreement (Jaccard similarity) between annotators")
# update colorbar
fig.update_coloraxes(colorscale="Reds")
# Hide y-axis tick labels
fig.update_yaxes(showticklabels=False)

fig.show()

# Print the average Jaccard similarity
print("Mean agreement (Jaccard similarity):")
print(jaccard_df.mean())  # FWI more agreement than output

#############################################################################################################

# General questions

# 30.2 = 5 mins on a single item
# 30.3 = ~45 minutes on average

# complexity of the task
print("Complexity of the task (mean):", data[data['Q30.2'].notna()]["Q30.4_1"].astype(int).mean())

# reason for the complexity:
    # almost everyone: difference between divergent and factually wrong information

# training helped?
print("Helpfulness of the training (mean):", data[data['Q30.2'].notna()]["Q30.6_1"].astype(int).mean())

# clarity of the guidelines
print("Clarity of the guidelines (mean):", data[data['Q30.2'].notna()]["Q30.7_1"].astype(int).mean())

# easiness of the UI
print("Easiness of the UI (mean):", data[data['Q30.2'].notna()]["Q30.10_1"].astype(int).mean())