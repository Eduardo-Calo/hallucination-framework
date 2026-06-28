import krippendorff
import numpy as np
import pandas as pd
import re


models = {
    "openai_gpt-5",
    # "openai_o3",
    "x-ai_grok-4",
    "anthropic_claude-sonnet-4",
    "anthropic_claude-opus-4.1",
    "google_gemini-2.5-pro",

    # "qwen_qwen3-235b-a22b-2507",
    "deepseek_deepseek-r1-0528",
    # "z-ai_glm-4.5-air",
    # "openai_gpt-oss-120b",
    # "moonshotai_kimi-k2",
    # "meta-llama_llama-4-maverick"
    } # Remember the underscore in the model names

models_dfs = {}
ambiguity_answers = {}

STRATEGY = "tot"  # specify the strategy used for analysis

# Check the strategy and load the appropriate data
if STRATEGY == "few_shot" or STRATEGY == "cot" or STRATEGY == "binary" or STRATEGY == "zero_shot":
    for model in models:
        try:
            df = pd.read_json(f"results/logic/results_{STRATEGY}_{model}.json")
            df = df.rename(columns={'predicted_category': model})
            df = df[['input', 'output', model]]
            models_dfs[model] = df
        except Exception as e:
            print(f"Error loading {model}: {e}")
elif STRATEGY == "tot":
    for model in models:
        try:
            df = pd.read_json(f"results/logic/results_{STRATEGY}_{model}.json")
            df = df.rename(columns={'predicted_category': model})
            df_step = df[['input', 'output', model, 'reasoning']]
            # Extract ambiguity answers
            ambiguity_answer = [df_step['reasoning'][i][0][1] for i in range(len(df_step))]
            ambiguity_answers[model] = ambiguity_answer
            # Continue with the usual processing
            df = df_step.drop(columns=['reasoning'], errors='ignore')
            models_dfs[model] = df
        except Exception as e:
            print(f"Error loading {model}: {e}")

# Merge all model DataFrames on input and output, preserving the order from the first model
first_model_df = next(iter(models_dfs.values()))
# Take rightmost column of all dfs
rightmost_columns = [df.iloc[:, -1] for df in models_dfs.values()]
rightmost_df = pd.concat(rightmost_columns, axis=1)
data = first_model_df[['input', 'output']].join(rightmost_df)

# Cast to numeric
def cast_str_to_int(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            contains_number = df[column].apply(lambda x: bool(re.search(r'^\d+$', str(x))))
            if contains_number.all():
                df[column] = df[column].astype(int)
    return df

############################################################################################################

# Load the gold data
data_gold = pd.read_csv("data/logic_gold.csv")

# Filter who didn't finish the experiment: drop rows with "Finished" == False
data_gold = data_gold[data_gold["Finished"] == "True"]

# Keep questions columns
data_gold = data_gold[[col for col in data_gold.columns if col.startswith("Q")]].iloc[:, 2:]

data_gold = cast_str_to_int(data_gold)

############################################################################################################

# map categories to numbers
categories = {
    "0a: Well-matched": 1,
    "1a: Output too weak": 3,
    "2a: Output too strong": 5,
    "2b: Output contradictory": 6,
    "3a: Input and Output independent": 7,
    "3b: Input and Output contradictory": 8,
}

# map letters to numbers
categories_letters = {
    "A": 1,
    "B": 3,
    "C": 5,
    "D": 6,
    "E": 7,
    "F": 8,
}

# reverse mapping
categories_reverse = {v: k for k, v in categories.items()}

# map categories to per step answers
per_step = {
    1: {"Q{item}.5": "No", "Q{item}.6": "Yes", "Q{item}.7": "Yes"},
    3: {"Q{item}.5": "No", "Q{item}.6": "Yes", "Q{item}.7": "No"},
    5: {"Q{item}.5": "No", "Q{item}.6": "No", "Q{item}.8": "Yes"},
    6: {"Q{item}.5": "Yes"},
    7: {"Q{item}.5": "No", "Q{item}.6": "No", "Q{item}.8": "No", "Q{item}.9": "No"},
    8: {"Q{item}.5": "No", "Q{item}.6": "No", "Q{item}.8": "No", "Q{item}.9": "Yes"}
}

############################################################################################################

passed_comprehension = data  # for consistency in naming

# Extract the category for each annotation

NUM_ITEMS = 75  # fixed number of items
START_BLOCK = 5  # start from 5 because the first 3 blocks are not questions and 4 is the comprehension check
GOLD_ANNOTATORS = 2  # 2 annotators in the gold data

# Reseting the index
data_gold = data_gold.reset_index(drop=True)

# Function to extract category for a specific annotation
def extract_category_from_annotation(df, item, annotator):

    global per_step, categories_reverse

    for category, questions in per_step.items():
        matching = all(df.loc[annotator, question.format(item=item)] == value for question, value in questions.items())
        if matching:
            return categories_reverse[category]
    
    return None

# Create a DataFrame to store the results
results_gold = pd.DataFrame(index=range(GOLD_ANNOTATORS))

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

if STRATEGY == "binary":
    results_final = pd.concat([passed_comprehension, results_gold.T.reset_index(drop=True).map(lambda x: categories_reverse.get(x, np.nan))], axis=1)
    # save results_final
    # results_final.to_csv("results/logic/results_binary.csv", index=False)
    # exit program
    exit()

results_crowd = passed_comprehension.drop(columns=['input', 'output'], errors='ignore').T.rename(columns=lambda x: f"Item {int(x) + 1}").map(lambda v: categories_letters.get(v, np.nan)).reset_index(drop=True)

# Calculate Krippendorff's alpha
print("Krippendorff's alpha for overall category:", krippendorff.alpha(reliability_data=results_crowd, level_of_measurement="nominal"))
# Calculate Krippendorff's alpha for gold data
print("Krippendorff's alpha for overall category (gold data):", krippendorff.alpha(reliability_data=results_gold, level_of_measurement="nominal"))

############################################################################################################

NUM_ANNOTATORS = len(results_crowd.index)

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

# retrieve first 2 votes and return as 2 columns
def first_two_votes(df):
    return df.value_counts().index[:2]

majority_gold = results_gold.apply(first_two_votes, axis=0).apply(lambda x: pd.Series(x))
# majority_gold = results_gold.mode().iloc[0]
# # add second column with NaNs
# majority_gold = majority_gold.to_frame()
# majority_gold[1] = np.nan

############################################################################################################

# accuracy of each annotator compared to the gold data
match_count_annotator = []
non_matching_items_annotator = []
for annotator in range(NUM_ANNOTATORS):
    match_count, non_matching_items = count_matches_annotator(results_crowd.loc[annotator], majority_gold)
    match_count_annotator.append(match_count)
    non_matching_items_annotator.append(non_matching_items)

# Print percentage and accuracies per model (use models from the model dictionary)
for idx, model in enumerate(models):
    accuracy = match_count_annotator[idx] / majority_gold.shape[0] * 100
    print(f"Model: {model} - Accuracy: {accuracy:.2f}% ({match_count_annotator[idx]}/{majority_gold.shape[0]})")

############################################################################################################

# analysis per category per llm

reports = []

for annotator in range(NUM_ANNOTATORS):

    pred_sets = results_crowd.loc[annotator].apply(lambda row: {row})
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
    reports.append(report_df)

# final report with categories on the top leftmost column and annotators on the top row
final_report = pd.concat(reports, keys=models, axis=1)

print(final_report)
# final_report.to_csv(f"results/logic/results_{STRATEGY}.csv")

############################################################################################################

# analysis per step per llm

if STRATEGY == "tot":
############################################################################################################

    # helper function for computing Krippendorff's alpha (handle perfect agreement)
    def compute_krippendorff_alpha(data):
        try:
            a = krippendorff.alpha(data, level_of_measurement="nominal")
        except AssertionError:
            a = 0
        return a

    i_o_answer = results_crowd.apply(lambda row: row.apply(lambda x: 1 if x in [1, 3] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 3: I -> O
    o_i_answer = results_crowd.apply(lambda row: row.apply(lambda x: 1 if x in [1, 2, 5] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1, 2 and 5: O -> I
    ambiguity_answer = pd.DataFrame(ambiguity_answers).T.map(lambda x: 1 if x == "yes" else (0 if x == "no" else np.nan))  # ambiguity answers
    ambiguity_answer.columns = results_crowd.columns
    ambiguity_answer.index = results_crowd.index

    # Calculate Krippendorff's alpha
    print("Krippendorff's alpha for I -> O:", compute_krippendorff_alpha(i_o_answer))
    print("Krippendorff's alpha for O -> I:", compute_krippendorff_alpha(o_i_answer))
    print("Krippendorff's alpha for ambiguity:", compute_krippendorff_alpha(ambiguity_answer))

############################################################################################################

    # accuracy on I -> O
    i_o_answer_gold = majority_gold.apply(lambda row: row.apply(lambda x: 1 if x in [1, 3] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1 and 3: I -> O

    # each annotator vs gold data
    match_count_annotator_i_o = []
    non_matching_items_annotator_i_o = []
    for annotator in range(NUM_ANNOTATORS):
        match_count, non_matching_items = count_matches_annotator(i_o_answer.loc[annotator], i_o_answer_gold)
        match_count_annotator_i_o.append(match_count)
        non_matching_items_annotator_i_o.append(non_matching_items)

############################################################################################################

    # accuracy on O -> I
    o_i_answer_gold = majority_gold.apply(lambda row: row.apply(lambda x: 1 if x in [1, 2, 5] else (0 if pd.notna(x) else np.nan)), axis=1)  # in categories 1, 2 and 5: O -> I

    # each annotator vs gold data
    match_count_annotator_o_i = []
    non_matching_items_annotator_o_i = []
    for annotator in range(NUM_ANNOTATORS):
        match_count, non_matching_items = count_matches_annotator(o_i_answer.loc[annotator], o_i_answer_gold)
        match_count_annotator_o_i.append(match_count)
        non_matching_items_annotator_o_i.append(non_matching_items)

############################################################################################################

    # accuracy on ambiguity
    ambiguous_answer_gold = data_gold[[f"Q{item}.3" for item in range(START_BLOCK, NUM_ITEMS + START_BLOCK)]].map(lambda x: 1 if x == "Yes" else (0 if x == "No" else np.nan))
    ambiguous_answer_gold.columns = results_crowd.columns
    ambiguous_answer_gold = ambiguous_answer_gold.T

    # each annotator vs gold data
    match_count_annotator_ambiguous = []
    non_matching_items_annotator_ambiguous = []
    for annotator in range(NUM_ANNOTATORS):
        match_count, non_matching_items = count_matches_annotator(ambiguity_answer.loc[annotator], ambiguous_answer_gold)
        match_count_annotator_ambiguous.append(match_count)
        non_matching_items_annotator_ambiguous.append(non_matching_items)

############################################################################################################

    df_results = pd.DataFrame({
        "Model": [model for model in models],
        "Overall": [match_count / NUM_ITEMS for match_count in match_count_annotator],
        "I -> O": [match_count / NUM_ITEMS for match_count in match_count_annotator_i_o],
        "O -> I": [match_count / NUM_ITEMS for match_count in match_count_annotator_o_i],
        "AMB": [match_count / NUM_ITEMS for match_count in match_count_annotator_ambiguous]
    })

    print(df_results)

############################################################################################################  