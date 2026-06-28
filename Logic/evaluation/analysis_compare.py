import pandas as pd
import numpy as np
import re
import krippendorff


# Load the data
data_crowd = pd.read_csv("evaluation/annotations/crowd_final.csv")
df = data_crowd[data_crowd['Progress'] == "100"]
df = df[df['Q1.2'] == "Yes"]

# Sort columns based on the numeric part of the column name
all_columns = df.columns.tolist()
q_columns = [col for col in all_columns if col.startswith('Q') and col != 'Q_RecaptchaScore']

# Extract the numeric part of each column name to sort, ignoring non-numeric suffixes
def extract_key(col_name):
    # Split at '.' and take the first part for primary sorting
    parts = col_name[1:].split('.')  # Skip 'Q'
    primary = int(parts[0])  # Convert the first part to an int
    secondary = int(parts[1].split('_')[0]) if len(parts) > 1 else 0  # Convert the second part to an int, or use 0
    return primary, secondary

q_columns_sorted = sorted(q_columns, key=extract_key)

ordered_df = df[['StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'Duration (in seconds)', 'Finished',
                  'RecordedDate', 'ResponseId', 'RecipientLastName', 'RecipientFirstName', 'RecipientEmail',
                  'ExternalReference', 'LocationLatitude', 'LocationLongitude', 'DistributionChannel',
                  'UserLanguage', 'Q_RecaptchaScore'] + q_columns_sorted + ['AnnotatorGroup']]

assert set(ordered_df.columns) == set(df.columns), "Columns are not the same"

# Keep questions columns
ordered_df = ordered_df[[col for col in ordered_df.columns if col.startswith("Q")]].iloc[:, 2:]

# Cast to numeric
def cast_str_to_int(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            contains_number = df[column].apply(lambda x: bool(re.search(r'^\d+$', str(x))))
            if contains_number.all():
                df[column] = df[column].astype(int)
    return df

ordered_df = cast_str_to_int(ordered_df)

passed_comprehension = ordered_df  # just to keep the same variable name as trivago experiment

############################################################################################################

# Load the gold data
data_gold = pd.read_csv("evaluation/gold_label/edu_kees.csv")
data_gold = data_gold[data_gold['Progress'] == "100"]
data_gold = data_gold[data_gold['Q1.2'] == "Yes"]

# Keep questions columns
data_gold = data_gold[[col for col in data_gold.columns if col.startswith("Q")]].iloc[:, 2:]
q_columns_sorted = sorted(data_gold, key=extract_key)

ordered_df_gold = data_gold[q_columns_sorted]

data_gold = cast_str_to_int(ordered_df_gold)

############################################################################################################

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

# Extract the category for each annotation

NUM_ANNOTATORS = passed_comprehension.shape[0]
NUM_ITEMS = 75  # fixed number of items
START_BLOCK = 5  # start from 5 because the first 4 blocks are not questions
ITEMS_PER_ANNOTATOR = 15
GOLD_ANNOTATORS = data_gold.shape[0]

# Reseting the index
passed_comprehension = passed_comprehension.reset_index(drop=True)
data_gold = data_gold.reset_index(drop=True)

############################################################################################################

# Function to extract category for a specific annotation
def extract_category_from_annotation(df, item, annotator):

    global per_step, categories_reverse

    for category, questions in per_step.items():
        matching = all(df.loc[annotator, question.format(item=item)] == value for question, value in questions.items())
        if matching:
            return categories_reverse[category]
    
    return None

# Create a DataFrame to store the results
results_crowd = pd.DataFrame(index=range(NUM_ANNOTATORS))
results_gold = pd.DataFrame(index=range(GOLD_ANNOTATORS))

# Loop through each item and each annotator, and store the results in the DataFrame
for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK):
    item_results = []
    for annotator in range(NUM_ANNOTATORS):
        category = extract_category_from_annotation(passed_comprehension, item, annotator)
        if category is not None:
            item_results.append(categories[category])
        else:
            item_results.append(np.nan)
    results_crowd[f"Item {item - (START_BLOCK - 1)}"] = item_results

# same for gold data
for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK):
    item_results = []
    for annotator in range(GOLD_ANNOTATORS):
        category = extract_category_from_annotation(data_gold, item, annotator)
        if category is not None:
            item_results.append(categories[category])
        else:
            item_results.append(np.nan)
    results_gold[f"Item {item - (START_BLOCK - 1)}"] = item_results

############################################################################################################

ambiguous_answer = passed_comprehension[[f"Q{item}.3" for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK)]].replace({"Yes": 1, "No": 0})
ambiguous_answer.columns = results_crowd.columns

results_gold_ambiguous = data_gold[[f"Q{item}.3" for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK)]].replace({"Yes": 1, "No": 0})
results_gold_ambiguous.columns = results_gold.columns

results_crowd_ambiguous = passed_comprehension[[f"Q{item}.3" for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK)]].replace({"Yes": 1, "No": 0})
results_crowd_ambiguous.columns = results_crowd.columns

############################################################################################################

# remove items 21, 23, 66, 67 (undefined)

items_to_remove = [21, 23, 66, 67]
blocks_to_remove = [item + START_BLOCK -1 for item in items_to_remove]

# if blocks_to_remove in column name then remove
passed_comprehension = passed_comprehension.drop(columns=[col for col in passed_comprehension.columns if any(f"Q{block}" in col for block in blocks_to_remove)])
data_gold = data_gold.drop(columns=[col for col in data_gold.columns if any(f"Q{block}" in col for block in blocks_to_remove)])

results_crowd = results_crowd[results_crowd.columns[~results_crowd.columns.isin([f"Item {item}" for item in items_to_remove])]]
results_gold = results_gold[results_gold.columns[~results_gold.columns.isin([f"Item {item}" for item in items_to_remove])]]

############################################################################################################

# retrieve majority votes
majority_crowd = results_crowd.mode()
majority_crowd = majority_crowd.stack().unstack(0)

# retrieve first 2 votes and return as 2 columns
def first_two_votes(df):
    return df.value_counts().index[:2]

majority_gold = results_gold.apply(first_two_votes, axis=0).apply(lambda x: pd.Series(x))
# majority_gold = results_gold.mode().iloc[0]
# # add second column with NaNs
# majority_gold = majority_gold.to_frame()
# majority_gold[1] = np.nan

num_items = majority_crowd.shape[0]

############################################################################################################

i_o_answer = results_crowd.apply(lambda row: row.apply(lambda x: 1 if x in [1, 3] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 3: I -> O
o_i_answer = results_crowd.apply(lambda row: row.apply(lambda x: 1 if x in [1, 5] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 5: O -> I

############################################################################################################

# compare majority vote with gold data
def count_matches(crowd_data, gold_data):
    # Convert Series to DataFrame if necessary
    if isinstance(crowd_data, pd.Series):
        crowd_data = crowd_data.to_frame()
    if isinstance(gold_data, pd.Series):
        gold_data = gold_data.to_frame()
    
    # Ensure both are DataFrames with the same index
    crowd_data = crowd_data.astype(float)  # Ensure numerical comparisons
    gold_data = gold_data.astype(float)
    
    # Initialize match count and list for non-matching items
    match_count = 0
    non_matching_items = []

    # Iterate through each row of the DataFrames
    for i, row in crowd_data.iterrows():
        # Get non-NaN values from crowd and gold data
        crowd_values = row.dropna().values
        gold_values = gold_data.loc[i].dropna().values
        
        # Check if any value in crowd_values matches any value in gold_values
        if any(item in gold_values for item in crowd_values):
            match_count += 1
        else:
            non_matching_items.append(i)

    return match_count, non_matching_items

def count_matches_annotator(crowd_data, gold_data):
    # Convert Series to DataFrame if necessary
    if isinstance(crowd_data, pd.Series):
        crowd_data = crowd_data.to_frame()
    if isinstance(gold_data, pd.Series):
        gold_data = gold_data.to_frame()
    
    # Ensure both are DataFrames with the same index
    crowd_data = crowd_data.astype(float)  # Ensure numerical comparisons
    gold_data = gold_data.astype(float)
    
    # Initialize match count and list for non-matching items
    match_count = 0
    non_matching_items = []

    # Iterate through each row of the DataFrames
    for i, row in crowd_data.iterrows():
        # Get non-NaN values from crowd and gold data
        crowd_values = row.dropna().values
        gold_values = gold_data.loc[i].dropna().values
        
        # Check if any value in crowd_values matches any value in gold_values
        if len(crowd_values) and len(gold_values):
            if any(item in gold_values for item in crowd_values):
                match_count += 1
            else:
                non_matching_items.append(i)

    return match_count, non_matching_items

############################################################################################################
# PAG@2: Majority vote(s) crowd over all gold vote(s)
# PAG@1: Majority vote(s) crowd over first gold vote
############################################################################################################

match_count_all, non_matching_items_all = count_matches(majority_crowd, majority_gold)
# print(f"Items where we disagree: {non_matching_items_all}")
match_count_first, non_matching_items_first = count_matches(majority_crowd, majority_gold[0])
# print(f"Items where we disagree: {non_matching_items_first}")

############################################################################################################

i_o_answer_gold = majority_gold.apply(lambda row: row.apply(lambda x: 1 if x in [1, 3] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 3: I -> O
i_o_answer_crowd = majority_crowd.apply(lambda row: row.apply(lambda x: 1 if x in [1, 3] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 3: I -> O

match_count_i_o, non_matching_items_i_o = count_matches(i_o_answer_crowd, i_o_answer_gold)
match_count_i_o_first, non_matching_items_i_o_first = count_matches(i_o_answer_crowd, i_o_answer_gold[0])

############################################################################################################

o_i_answer_gold = majority_gold.apply(lambda row: row.apply(lambda x: 1 if x in [1, 5] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 5: O -> I
o_i_answer_crowd = majority_crowd.apply(lambda row: row.apply(lambda x: 1 if x in [1, 5] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 5: O -> I

match_count_o_i, non_matching_items_o_i = count_matches(o_i_answer_crowd, o_i_answer_gold)
match_count_o_i_first, non_matching_items_o_i_first = count_matches(o_i_answer_crowd, o_i_answer_gold[0])

############################################################################################################

# collapse to 1 row. if at least 1 annotator says yes, then it is ambiguous
results_gold_ambiguous = results_gold_ambiguous.any()
results_crowd_ambiguous = results_crowd_ambiguous.any()

ambiguous_answer_gold = pd.DataFrame({0: results_gold_ambiguous.astype(int), 1: float('nan')}, index=results_gold_ambiguous.index)
ambiguous_answer_crowd = pd.DataFrame({0: results_crowd_ambiguous.astype(int), 1: float('nan')}, index=results_crowd_ambiguous.index)

match_count_ambiguous, non_matching_items_ambiguous = count_matches(results_crowd_ambiguous, results_gold_ambiguous)
match_count_ambiguous_first, non_matching_items_ambiguous_first = count_matches(results_crowd_ambiguous, results_gold_ambiguous)

############################################################################################################

# print table with results

pag_results = pd.DataFrame({
    "PAG@2": [match_count_all, match_count_i_o, match_count_o_i, match_count_ambiguous],
    "PAG@2 %": [match_count_all / num_items, match_count_i_o / num_items, match_count_o_i / num_items, match_count_ambiguous / num_items],
    "PAG@1": [match_count_first, match_count_i_o_first, match_count_o_i_first, match_count_ambiguous_first],
    "PAG@1 %": [match_count_first / num_items, match_count_i_o_first / num_items, match_count_o_i_first / num_items, match_count_ambiguous_first / num_items],
    "Items": [num_items, num_items, num_items, num_items],
}, index=["Overall", "I -> O", "O -> I", "Ambiguity"])

print(pag_results)

############################################################################################################
# WITH BAD ANNOTATORS
print("APA ALL") # any match with gold counts
############################################################################################################

# IAA (all)

print("Krippendorff's alpha overall (all):", krippendorff.alpha(reliability_data=results_crowd, level_of_measurement='nominal'))
print("Krippendorff's alpha I -> O (all):", krippendorff.alpha(reliability_data=i_o_answer, level_of_measurement='nominal'))
print("Krippendorff's alpha O -> I (all):", krippendorff.alpha(reliability_data=o_i_answer, level_of_measurement='nominal'))
print("Krippendorff's alpha ambiguity (all):", krippendorff.alpha(reliability_data=ambiguous_answer, level_of_measurement='nominal'))

############################################################################################################

# overall

num_answers_per_annotator = results_crowd.count(axis=1)

# compare each annotator with gold data
match_count_annotator = []
non_matching_items_annotator = []
for annotator in range(NUM_ANNOTATORS):
    match_count, non_matching_items = count_matches_annotator(results_crowd.loc[annotator], majority_gold)
    match_count_annotator.append(match_count)
    non_matching_items_annotator.append(non_matching_items)

# print results
# print("Majority vote(s) crowd over gold vote(s) for each annotator:")
# for annotator, (match_count, non_matching_items) in enumerate(zip(match_count_annotator, non_matching_items_annotator)):
#     print(f"Annotator {annotator + 1}:")
#     print(f"Items where we disagree: {non_matching_items}")
#     print(f"Number of items: {15}")
#     print(f"Number of matches: {match_count}")
#     print(f"Percentage of matches: {match_count / 15 * 100:.2f}%")
#     print()

# mean accuracy over gold data
print(f"APA Gold (Overall): {np.mean(match_count_annotator / num_answers_per_annotator):.2f}")
# std
print(f"APA Gold (Overall) std: {np.std(match_count_annotator / num_answers_per_annotator):.2f}")

# majority crowd vs each annotator
match_count_annotator_majority = []
non_matching_items_annotator_majority = []
for annotator in range(NUM_ANNOTATORS):
    match_count, non_matching_items = count_matches_annotator(results_crowd.loc[annotator], majority_crowd)
    match_count_annotator_majority.append(match_count)
    non_matching_items_annotator_majority.append(non_matching_items)

# print results
# print("Majority vote(s) crowd over each annotator:")
# for annotator, (match_count, non_matching_items) in enumerate(zip(match_count_annotator_majority, non_matching_items_annotator_majority)):
#     print(f"Annotator {annotator + 1}:")
#     print(f"Items where they disagree: {non_matching_items}")
#     print(f"Number of items: {15}")
#     print(f"Number of matches: {match_count}")
#     print(f"Percentage of matches: {match_count / 15 * 100:.2f}%")
#     print()

# mean accuracy over majority crowd
print(f"APA Crowd (Overall): {np.mean(match_count_annotator_majority / num_answers_per_annotator):.2f}")

############################################################################################################

# I -> O

# each annotator vs gold data
match_count_annotator_i_o = []
non_matching_items_annotator_i_o = []
for annotator in range(NUM_ANNOTATORS):
    match_count, non_matching_items = count_matches_annotator(i_o_answer.loc[annotator], i_o_answer_gold)
    match_count_annotator_i_o.append(match_count)
    non_matching_items_annotator_i_o.append(non_matching_items)

# print results
# print("Majority vote(s) crowd over gold vote(s) for each annotator (I -> O):")
# for annotator, (match_count, non_matching_items) in enumerate(zip(match_count_annotator_i_o, non_matching_items_annotator_i_o)):
#     print(f"Annotator {annotator + 1}:")
#     print(f"Items where we disagree: {non_matching_items}")
#     print(f"Number of items: {15}")
#     print(f"Number of matches: {match_count}")
#     print(f"Percentage of matches: {match_count / 15 * 100:.2f}%")
#     print()

# mean accuracy over gold data
print(f"APA Gold (I -> O): {np.mean(match_count_annotator_i_o / num_answers_per_annotator):.2f}")
# std
print(f"APA Gold (I -> O) std: {np.std(match_count_annotator_i_o / num_answers_per_annotator):.2f}")

# majority crowd vs each annotator
match_count_annotator_majority_i_o = []
non_matching_items_annotator_majority_i_o = []
for annotator in range(NUM_ANNOTATORS):
    match_count, non_matching_items = count_matches_annotator(i_o_answer.loc[annotator], i_o_answer_crowd)
    match_count_annotator_majority_i_o.append(match_count)
    non_matching_items_annotator_majority_i_o.append(non_matching_items)

# print results
# print("Majority vote(s) crowd over each annotator (I -> O):")
# for annotator, (match_count, non_matching_items) in enumerate(zip(match_count_annotator_majority_i_o, non_matching_items_annotator_majority_i_o)):
#     print(f"Annotator {annotator + 1}:")
#     print(f"Items where they disagree: {non_matching_items}")
#     print(f"Number of items: {15}")
#     print(f"Number of matches: {match_count}")
#     print(f"Percentage of matches: {match_count / 15 * 100:.2f}%")
#     print()

# mean accuracy over majority crowd
print(f"APA Crowd (I -> O): {np.mean(match_count_annotator_majority_i_o / num_answers_per_annotator):.2f}")

############################################################################################################

# O -> I

# each annotator vs gold data
match_count_annotator_o_i = []
non_matching_items_annotator_o_i = []
for annotator in range(NUM_ANNOTATORS):
    match_count, non_matching_items = count_matches_annotator(o_i_answer.loc[annotator], o_i_answer_gold)
    match_count_annotator_o_i.append(match_count)
    non_matching_items_annotator_o_i.append(non_matching_items)

# print results
# print("Majority vote(s) crowd over gold vote(s) for each annotator (O -> I):")
# for annotator, (match_count, non_matching_items) in enumerate(zip(match_count_annotator_o_i, non_matching_items_annotator_o_i)):
#     print(f"Annotator {annotator + 1}:")
#     print(f"Items where we disagree: {non_matching_items}")
#     print(f"Number of items: {15}")
#     print(f"Number of matches: {match_count}")
#     print(f"Percentage of matches: {match_count / 15 * 100:.2f}%")
#     print()

# mean accuracy over gold data
print(f"APA Gold (O -> I): {np.mean(match_count_annotator_o_i / num_answers_per_annotator):.2f}")
# std
print(f"APA Gold (O -> I) std: {np.std(match_count_annotator_o_i / num_answers_per_annotator):.2f}")

# majority crowd vs each annotator
match_count_annotator_majority_o_i = []
non_matching_items_annotator_majority_o_i = []
for annotator in range(NUM_ANNOTATORS):
    match_count, non_matching_items = count_matches_annotator(o_i_answer.loc[annotator], o_i_answer_crowd)
    match_count_annotator_majority_o_i.append(match_count)
    non_matching_items_annotator_majority_o_i.append(non_matching_items)

# print results
# print("Majority vote(s) crowd over each annotator (O -> I):")
# for annotator, (match_count, non_matching_items) in enumerate(zip(match_count_annotator_majority_o_i, non_matching_items_annotator_majority_o_i)):
#     print(f"Annotator {annotator + 1}:")
#     print(f"Items where they disagree: {non_matching_items}")
#     print(f"Number of items: {15}")
#     print(f"Number of matches: {match_count}")
#     print(f"Percentage of matches: {match_count / 15 * 100:.2f}%")
#     print()

# mean accuracy over majority crowd
print(f"APA Crowd (O -> I): {np.mean(match_count_annotator_majority_o_i / num_answers_per_annotator):.2f}")

############################################################################################################

# ambiguity

# each annotator vs gold data
match_count_annotator_ambiguous = []
non_matching_items_annotator_ambiguous = []
for annotator in range(NUM_ANNOTATORS):
    match_count, non_matching_items = count_matches_annotator(ambiguous_answer.loc[annotator], ambiguous_answer_gold)
    match_count_annotator_ambiguous.append(match_count)
    non_matching_items_annotator_ambiguous.append(non_matching_items)

# # print results
# print("Majority vote(s) crowd over gold vote(s) for each annotator (ambiguity):")
# for annotator, (match_count, non_matching_items) in enumerate(zip(match_count_annotator_ambiguous, non_matching_items_annotator_ambiguous)):
#     print(f"Annotator {annotator + 1}:")
#     print(f"Items where we disagree: {non_matching_items}")
#     print(f"Number of items: {15}")
#     print(f"Number of matches: {match_count}")
#     print(f"Percentage of matches: {match_count / 15 * 100:.2f}%")
#     print()

# mean accuracy over gold data
print(f"APA Gold (Ambiguity): {np.mean(match_count_annotator_ambiguous / num_answers_per_annotator):.2f}")
# std
print(f"APA Gold (Ambiguity) std: {np.std(match_count_annotator_ambiguous / num_answers_per_annotator):.2f}")

# majority crowd vs each annotator
match_count_annotator_majority_ambiguous = []
non_matching_items_annotator_majority_ambiguous = []
for annotator in range(NUM_ANNOTATORS):
    match_count, non_matching_items = count_matches_annotator(ambiguous_answer.loc[annotator], ambiguous_answer_crowd)
    match_count_annotator_majority_ambiguous.append(match_count)
    non_matching_items_annotator_majority_ambiguous.append(non_matching_items)

# print results
# print("Majority vote(s) crowd over each annotator (ambiguity):")
# for annotator, (match_count, non_matching_items) in enumerate(zip(match_count_annotator_majority_ambiguous, non_matching_items_annotator_majority_ambiguous)):
#     print(f"Annotator {annotator + 1}:")
#     print(f"Items where they disagree: {non_matching_items}")
#     print(f"Number of items: {15}")
#     print(f"Number of matches: {match_count}")
#     print(f"Percentage of matches: {match_count / 15 * 100:.2f}%")
#     print()

# mean accuracy over majority crowd
print(f"APA Crowd (Ambiguity): {np.mean(match_count_annotator_majority_ambiguous / num_answers_per_annotator):.2f}")

# ############################################################################################################

# analysis per category

pred_sets = majority_crowd.apply(lambda row: set(row.dropna().astype(int)), axis=1)
true_sets = majority_gold.apply(lambda row: set(row.dropna().astype(int)), axis=1)

all_labels = sorted(set.union(*pred_sets, *true_sets))

# Compute per-label stats
results = []
for label in all_labels:
    tp = sum(label in p and label in t for p, t in zip(pred_sets, true_sets))
    fp = sum(label in p and label not in t for p, t in zip(pred_sets, true_sets))
    fn = sum(label not in p and label in t for p, t in zip(pred_sets, true_sets))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    support   = sum(label in t for t in true_sets)
    pred_ct   = sum(label in p for p in pred_sets)

    results.append({
        'label': categories_reverse.get(label, str(label)),
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': support,
        'pred_count': pred_ct
    })

macro = {
    'label': 'macro avg',
    'precision': np.mean([r['precision'] for r in results]),
    'recall': np.mean([r['recall'] for r in results]),
    'f1-score': np.mean([r['f1-score'] for r in results]),
    'support': sum(r['support'] for r in results),
    'pred_count': sum(r['pred_count'] for r in results)
}
results.append(macro)

report_df = pd.DataFrame(results).set_index("label").round(2)
print(report_df)