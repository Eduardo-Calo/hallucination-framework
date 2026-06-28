"""Python script to extract all correct FOL formulae from the
Grade Grinder Corpus, with several preprocessing steps."""

import pandas as pd
import re

# Open GGC as DataFrame
ggc_df = pd.read_csv("../data/translationcorpus-1.0.1.csv", 
                     header=0, 
                     on_bad_lines="skip")
print("original number of translations",len(ggc_df))
# --> 19348162


sentnum_df = pd.read_csv("../data/sentences.csv", header=None, names=["exnum", "sent"], usecols=[1, 2])
sentnum = dict(zip(sentnum_df.exnum, sentnum_df.sent))

# These exnums have doblet semantically different golden references
doblets = set([x for x in list(sentnum_df.exnum) if list(sentnum_df.exnum).count(x) > 1])


# Take the canonical form of the formulas that were correct
ggc_df = ggc_df[(ggc_df["status"] == "correct")][["exsentnum", "canonical"]].drop_duplicates()
formulas = ggc_df[~(ggc_df.exsentnum.isin(doblets))] # remove doblets lines
print("initial amount of correct formulas:", len(formulas))
# --> 25522


# Remove NaNs
formulas = formulas.dropna()
print("after removing nans:", len(formulas))
# --> 25455


# Remove formulas with these specific characters or strings: FOL with equality, n-ary predicates, canonical connectives, quantifiers
ignore = "|".join(["<", ">", "\^", "\+", "\*", "%", "is", "not", "equivalent", "\.", "\"", "\\", "0", "1", "2", "3", "4", "5", "\:"])
formulas = formulas.loc[~formulas.canonical.str.contains(ignore), :]

# Remove formulas with these predicates. only geometric domain
predicates = ["Pet", "Student", "Gave", "Person", "Owned", "Prime"]
formulas = formulas.loc[~formulas.canonical.str.contains("|".join(predicates)), :]
print("after removing formulas with specific characters or predicates:", len(formulas))

# Remove formulas with no predicates
formulas_no_preds = formulas.loc[~formulas.canonical.str.contains(r'[A-Z]'), :]
formulas = formulas[~formulas["canonical"].isin(formulas_no_preds["canonical"])]
print("after removing formulas with no predicates:", len(formulas))



# Change to correct spacing:
# - Put spaces before and after the given characters: ,|&$()@/~
formulas.canonical = [re.sub(r'([,|&$()@/~])', " \\1 ", f) for f in formulas.canonical]
# - Remove double spaces
formulas.canonical = [re.sub(' +', ' ', f) for f in formulas.canonical]
# - Remove beginning and end spaces
formulas.canonical = [f[1:] if f[0]==' ' else f for f in formulas.canonical]
formulas.canonical = [f[:-1] if f[-1]==' ' else f for f in formulas.canonical]


# Replace exsentnum with gold ref and rename
formulas.exsentnum = formulas.exsentnum.apply(lambda x: sentnum[x])
formulas.rename(columns={"exsentnum": "reference", "canonical": "formula"}, inplace=True)
formulas.sort_values("reference", inplace=True)


# calculate statistics on formulas #
# histogram of formula lengths (predicates, connectives, quantifiers, variables)
formulas["length"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c not in ["(", ")", ","]]))
print(formulas["length"].describe())
# number of predicates per formula
formulas["npreds"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c[0].isupper()]))
print(formulas["npreds"].describe())
# number of connectives per formula
formulas["conns"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c in ["&", "|", "~", "$"]]))
print(formulas["conns"].describe())
# number of quantifiers per formula
formulas["quants"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c in ["@", "/"]]))
print(formulas["quants"].describe())
# numbers of negations per formula
formulas["negs"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c == "~"]))
print(formulas["negs"].describe())
# number of conjunctions per formula
formulas["ands"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c == "&"]))
print(formulas["ands"].describe())
# number of disjunctions per formula
formulas["ors"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c == "|"]))
print(formulas["ors"].describe())
# number of implications per formula
formulas["imps"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c == "$"]))
print(formulas["imps"].describe())
# number of universal quantifiers per formula
formulas["uquants"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c == "@"]))
print(formulas["uquants"].describe())
# number of existential quantifiers per formula
formulas["equants"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c == "/"]))
print(formulas["equants"].describe())
# number of variables per formula
formulas["vars"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c.islower()]))
print(formulas["vars"].describe())
# number of = per formula
formulas["eqs"] = formulas["formula"].apply(lambda x: len([c for c in x.split() if c == "="]))
print(formulas["eqs"].describe())
# count of each predicate
formulas["preds"] = formulas["formula"].apply(lambda x: re.findall(r'[A-Z][A-Za-z]+\s\([a-z\,\s]+\)', x))
# unary predicates
formulas["unary"] = formulas["preds"].apply(lambda x: len([p for p in x if p.count(",") == 0]))
print(formulas["unary"].describe())
# binary predicates
formulas["binary"] = formulas["preds"].apply(lambda x: len([p for p in x if p.count(",") == 1]))
print(formulas["binary"].describe())
# ternary predicates
formulas["ternary"] = formulas["preds"].apply(lambda x: len([p for p in x if p.count(",") == 2]))
print(formulas["ternary"].describe())
# max is 3-ary predicates


# sample of formulas (remove those with similar characteristics/structure)
formulas.drop_duplicates(subset=["reference", "length", "npreds", "conns", "quants", "negs", "ands", "ors", "imps", "uquants", "equants", "vars", "eqs", "unary", "binary", "ternary"], inplace=True)

# describe the dataset
print(formulas.describe())
formulas.to_csv("../data/dataset.csv", index=False)

# pick representative varied sample of formulas based on length, npreds, etc.
sample = formulas.sample(50, random_state=42)
sample.to_csv("data/final_sample.csv", index=False)
print(sample.describe())

# compare the 2 means
formulas.describe().loc["mean"] - sample.describe().loc["mean"]




# generate text

import pandas as pd
import requests

def preprocess_formula(formula):
    formula = formula.replace("~", "¬")
    formula = formula.replace("&", "∧")
    formula = formula.replace("|", "∨")
    formula = formula.replace("$", "→")
    formula = formula.replace("@", "∀")
    formula = formula.replace("/", "∃")
    return formula

preds = set([(x.split()[0], x.count(",")) for x in [item for sublist in formulas["preds"].tolist() for item in sublist]])
# {('SameSize', 1), ('Smaller', 1), ('SameCol', 1), ('Larger', 1), ('Medium', 0), ('Large', 0), ('FrontOf', 1), ('Adjoins', 1), ('Small', 0), ('Between', 2), ('LeftOf', 1), ('Cube', 0), ('Dodec', 0), ('RightOf', 1), ('SameRow', 1), ('SameShape', 1), ('Tet', 0), ('BackOf', 1)}

# predicates explained
exp = """
SameSize ( x , y ): x and y are the same size.
Smaller ( x , y ): x is smaller than y.
SameCol ( x , y ): x and y are in the same column.
Larger ( x , y ): x is larger than y.
BackOf ( x , y ): x is behind y.
Medium ( x ): x is medium.
Large ( x ): x is large.
FrontOf ( x , y ): x is in front of y.
Adjoins ( x , y ): x adjoins y.
Small ( x ): x is small.
Between ( x , y , z ): x is between y and z.
LeftOf ( x , y ): x is to the left of y.
Cube ( x ): x is a cube.
Dodec ( x ): x is a dodecahedron.
RightOf ( x , y ): x is to the right of y.
SameRow ( x , y ): x and y are in the same row.
SameShape ( x , y ): x and y are the same shape.
Tet ( x ): x is a tetrahedron.
"""

API_TOKEN = ""

def model_inference(model, datapoint, token=API_TOKEN):
    model = model
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}

    parameters = {"max_new_tokens": 250, "temperature": 0}  # "return_full_text": False, 
    options = {"wait_for_model": True}  # set to True if you want to wait for the model to be loaded

    def query_model(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    input_data = datapoint
    prompt = f"Translate the following formula into English.\nThe following is the meaning of the predicates used in the formula:\n{exp}\nONLY RETURN THE TRANSLATION. DO NOT USE LOGICAL SYMBOLS. DO NOT GIVE ANY EXPLANATION.\n\nFormula: {input_data}\nTranslation: "

    output = query_model(
        {
            "inputs": prompt,
            "parameters": parameters,
            "options": options
        }
    )

    print(output)

    try:
        translation = output[0]["generated_text"]
        translation = translation.strip().replace("\n", " ")
    except:
        translation = output

    return translation

# ["togethercomputer/Llama-2-7B-32K-Instruct", # too big
# "codellama/CodeLlama-34b-Instruct-hf", # ok
# "mistralai/Mixtral-8x7B-Instruct-v0.1", # ok
# "tiiuae/falcon-7b-instruct", # bad
# "bigscience/bloom-1b7", # busy
# "google/flan-t5-(x)xl"] # bad

models = ["codellama/CodeLlama-34b-Instruct-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
data = [preprocess_formula(f) for f in sample["formula"].tolist()]


final_data = []

for formula in data:
    for model in models:
        input = formula
        output = model_inference(model, formula, token=API_TOKEN)
        final_data.append((input, output, model))


df_final_data = pd.DataFrame(final_data, columns=["input", "output", "model"])
df_final_data.to_csv("data/input_output.csv", index=False)

# clean codellama outputs
# ["output"].apply(lambda x: [x.split("Answer: ")[1] if "Answer" in x and isinstance(x, str) else x][0])





# gemini inference

import google.generativeai as genai # needs new protobuf (update if needed)
import time

genai.configure(api_key="")
gemini = genai.GenerativeModel("gemini-1.0-pro")

sample = pd.read_csv("data/final_sample.csv")
data = [preprocess_formula(f) for f in sample["formula"].tolist()]

exp = """
SameSize ( x , y ): x and y are the same size.
Smaller ( x , y ): x is smaller than y.
SameCol ( x , y ): x and y are in the same column.
Larger ( x , y ): x is larger than y.
BackOf ( x , y ): x is behind y.
Medium ( x ): x is medium.
Large ( x ): x is large.
FrontOf ( x , y ): x is in front of y.
Adjoins ( x , y ): x adjoins y.
Small ( x ): x is small.
Between ( x , y , z ): x is between y and z.
LeftOf ( x , y ): x is to the left of y.
Cube ( x ): x is a cube.
Dodec ( x ): x is a dodecahedron.
RightOf ( x , y ): x is to the right of y.
SameRow ( x , y ): x and y are in the same row.
SameShape ( x , y ): x and y are the same shape.
Tet ( x ): x is a tetrahedron.
"""

def gemini_inference(input_data):
    prompt = f"Translate the following formula into English.\nThe following is the meaning of the predicates used in the formula:\n{exp}\nONLY RETURN THE TRANSLATION. DO NOT USE LOGICAL SYMBOLS. DO NOT GIVE ANY EXPLANATION.\n\nFormula: {input_data}\nTranslation: "
    response = gemini.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0)) # deterministic
    print(response.text)
    return response.text

final_data = []
delay = 60 / 15  # 4 seconds delay

for formula in data:
    input_data = formula
    output = gemini_inference(formula)
    final_data.append((input_data, output, "gemini-1.0-pro"))

    time.sleep(delay)  # Wait for 4 seconds between each request

df_gemini_data = pd.DataFrame(final_data, columns=["input", "output", "model"])
df_gemini_data.to_csv("data/gemini_input_output.csv", index=False)





# gpt inference
from openai import OpenAI

client = OpenAI(api_key="")

def gpt_inference(input_data):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Translate the following formula into English.\nThe following is the meaning of the predicates used in the formula:\n{exp}\nONLY RETURN THE TRANSLATION. DO NOT USE LOGICAL SYMBOLS. DO NOT GIVE ANY EXPLANATION.\n\nFormula: {input_data}\nTranslation: ",
            }
        ],
        model="gpt-3.5-turbo",
        temperature=0,
    )
    print(chat_completion)
    return chat_completion.choices[0].message.content

final_data = []

for formula in data:
    input_data = formula
    output = gpt_inference(formula)
    final_data.append((input_data, output, "gpt-3.5-turbo"))

df_gpt_data = pd.DataFrame(final_data, columns=["input", "output", "model"])
df_gpt_data.to_csv("data/gpt_input_output.csv", index=False)





# bloom inference
# bloomz-3b sucks
# phi is good

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

def preprocess_formula(formula):
    formula = formula.replace("~", "¬")
    formula = formula.replace("&", "∧")
    formula = formula.replace("|", "∨")
    formula = formula.replace("$", "→")
    formula = formula.replace("@", "∀")
    formula = formula.replace("/", "∃")
    return formula

exp = """
SameSize ( x , y ): x and y are the same size.
Smaller ( x , y ): x is smaller than y.
SameCol ( x , y ): x and y are in the same column.
Larger ( x , y ): x is larger than y.
BackOf ( x , y ): x is behind y.
Medium ( x ): x is medium.
Large ( x ): x is large.
FrontOf ( x , y ): x is in front of y.
Adjoins ( x , y ): x adjoins y.
Small ( x ): x is small.
Between ( x , y , z ): x is between y and z.
LeftOf ( x , y ): x is to the left of y.
Cube ( x ): x is a cube.
Dodec ( x ): x is a dodecahedron.
RightOf ( x , y ): x is to the right of y.
SameRow ( x , y ): x and y are in the same row.
SameShape ( x , y ): x and y are the same shape.
Tet ( x ): x is a tetrahedron.
"""

sample = pd.read_csv("data/final_sample.csv")
data = [preprocess_formula(f) for f in sample["formula"].tolist()]


checkpoint = "microsoft/Phi-3.5-mini-instruct"

model = pipeline(model=checkpoint, model_kwargs={"max_length": 2000, "temperature": 0.0})

def phi_inference(input_data):
    prompt = f"Translate the following formula into English.\nThe following is the meaning of the predicates used in the formula:\n{exp}\nONLY RETURN THE TRANSLATION. DO NOT USE LOGICAL SYMBOLS. DO NOT GIVE ANY EXPLANATION.\n\nFormula: {input_data}\nTranslation: "
    outputs = model(prompt, return_full_text=False)
    return outputs[0]["generated_text"]

final_data = []

for formula in tqdm(data):
    input_data = formula
    output = phi_inference(formula)
    print(output)
    final_data.append((input_data, output.strip(), "microsoft/Phi-3.5-mini-instruct"))

df_phi_data = pd.DataFrame(final_data, columns=["input", "output", "model"])
df_phi_data.to_csv("data/phi_input_output.csv", index=False)
