# ==================================
# 1. Imports and Setup
# ==================================
import json
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==================================
# 2. Constants for Prompting Strategies
# ==================================

# --- Strategy 0: Binary Classification ---
BINARY_EXAMPLES = [
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is a 5-star hotel in Mexico.", "category": "YES"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is an incredibly charming 5-star hotel in Mexico.", "category": "NO"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is a hotel in Mexico.", "category": "NO"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is an incredibly charming hotel in Mexico.", "category": "NO"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel", "Hotel facilities": "Lockers"}, "output": "Luna Blanca is a 5-star hotel in Mexico that provides lockers and a swimming pool.", "category": "NO"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is a 5-star hotel in New York, Mexico.", "category": "NO"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel", "Hotel facilities": "Lockers"}, "output": "Luna Blanca is a 5-star hotel in Mexico that provides a swimming pool.", "category": "NO"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is a 5-star hotel in the USA.", "category": "NO"}
]

BINARY_CLASSIFICATION_PROMPT = """You are participating in an experiment.
You will see an Input that is a set of structured facts and an Output that is its natural language description.
Your task is to CLASSIFY the nature of the logical relationship between Input and Output.

LOGICAL CONSEQUENCE:
- Input entails Output if and only if by reading Input, you can infer that Output is true.
- Output entails Input if and only if by reading Output, you can infer that Input is true.

Ignore typos or grammar mistakes in Output.

Answer YES if Input entails Output and Output entails Input. Otherwise, answer NO. Do not add explanations.

Examples:
"""

# --- Shared Examples for Few-Shot and CoT ---
EXAMPLES = [
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is a 5-star hotel in Mexico.", "category": "A"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is an incredibly charming 5-star hotel in Mexico.", "category": "B"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is a hotel in Mexico.", "category": "C"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is an incredibly charming hotel in Mexico.", "category": "D"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel", "Hotel facilities": "Lockers"}, "output": "Luna Blanca is a 5-star hotel in Mexico that provides lockers and a swimming pool.", "category": "E"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is a 5-star hotel in New York, Mexico.", "category": "F"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel", "Hotel facilities": "Lockers"}, "output": "Luna Blanca is a 5-star hotel in Mexico that provides a swimming pool.", "category": "G"},
    {"input": {"Name": "Luna Blanca", "Star rating": 5, "Country": "Mexico", "Accommodation type": "Hotel"}, "output": "Luna Blanca is a 5-star hotel in the USA.", "category": "H"}
]


# --- Strategy 0.1: 0-Shot ---
ZERO_SHOT_BASE_PROMPT = """You are participating in an experiment.
You will see an Input that is a set of structured facts and an Output that is its natural language description.
Your task is to CLASSIFY the nature of the logical relationship between Input and Output.

Ignore typos or grammar mistakes in Output.

Only return the capital letter corresponding to the correct category. Do not add explanations.

Possible Categories:
A: Input and Output are well-matched
B: Input and Output are well-matched, with Output containing harmless additions
C: Output is too weak with respect to Input
D: Output is too weak with respect to Input, with Output containing harmless additions
E: Output is too strong with respect to Input
F: Output is self-contradictory
G: Input and Output are logically independent of each other
H: Input and Output contradict each other
"""


# --- Strategy 1: Few-Shot ---
FEW_SHOT_BASE_PROMPT = """You are participating in an experiment.
You will see an Input that is a set of structured facts and an Output that is its natural language description.
Your task is to CLASSIFY the nature of the logical relationship between Input and Output.

Ignore typos or grammar mistakes in Output.

Only return the capital letter corresponding to the correct category. Do not add explanations.

Possible Categories:
A: Input and Output are well-matched
B: Input and Output are well-matched, with Output containing harmless additions
C: Output is too weak with respect to Input
D: Output is too weak with respect to Input, with Output containing harmless additions
E: Output is too strong with respect to Input
F: Output is self-contradictory
G: Input and Output are logically independent of each other
H: Input and Output contradict each other

Examples:
"""


# --- Strategy 2: Chain-of-Thought (CoT) ---
def _generate_cot_examples_text(examples: List[Dict]) -> str:
    """Generates the detailed examples section for the CoT prompt."""
    categories = "ABCDEFGH"
    full_text = []

    description_map = {
        'A': "Input and Output are well-matched",
        'B': "Input and Output are well-matched, with Output containing harmless additions",
        'C': "Output is too weak with respect to Input",
        'D': "Output is too weak with respect to Input, with Output containing harmless additions",
        'E': "Output is too strong with respect to Input",
        'F': "Output is self-contradictory",
        'G': "Input and Output are logically independent of each other",
        'H': "Input and Output contradict each other"
    }

    reasoning_map = {
        'A': "Input entails Output, and Output entails Input.",
        'B': "Input does not entail Output, and Output entails Input, but the additional information in Output is not factually wrong.",
        'C': "Input entails Output, and Output does not entail Input.",
        'D': "Input does not entail Output, and Output does not entail Input, and the additional information in Output is not factually wrong.",
        'E': "Input does not entail Output, Output entails Input, and the additional information in Output is factually wrong.",
        'F': "Output contradicts itself, i.e., it contains statements that cannot be true at the same time.",
        'G': "Input does not entail Output, and Output does not entail Input, and the additional information in Output is factually wrong, but Output does not contradict Input.",
        'H': "Input does not entail Output, Output does not entail Input, and the additional information in Output is factually wrong, and Output contradicts Input."
    }

    for cat in categories:
        ex = next((e for e in examples if e['category'] == cat), None)
        if not ex: continue
        input_str = "\n".join([f"{k}: {v}" for k, v in ex['input'].items()])
        example_text = f"Example:\nInput:\n{input_str}\nOutput:\n{ex['output']}\n"
        example_text += f"REASONING: {reasoning_map[cat]}\n"
        example_text += f"CATEGORY {cat}: {description_map[cat]}" 
        full_text.append(example_text)
        
    return "\n\n".join(full_text)


COT_BASE_PROMPT_TEMPLATE = """You are participating in an experiment.
You will see an Input that is a set of structured facts and an Output that is its natural language description.
Your task is to analyze Output in light of Input based on the relation of LOGICAL CONSEQUENCE, and to CLASSIFY the nature of the logical relationship between them.

----------------------------------------------------

DEFINITIONS:

LOGICAL CONSEQUENCE:
- Input entails Output if and only if by reading Input, you can infer that Output is true.
- Output entails Input if and only if by reading Output, you can infer that Input is true.

FACTUALLY WRONG INFORMATION:
When Input does not entail Output, Output contains information that is not in Input.
This information in Output may or may not be factually wrong.
We define as factually wrong any information in Output that conveys facts that could well turn out to be wrong, given the information in Input.
Factually wrong information does NOT comprise: (i) subjective opinions, (ii) vague or ambiguous statements, (iii) information that is unverifiable, (iv) information that can be inferred from Input to hold in most (but not all) cases.

----------------------------------------------------

Following are EXAMPLES for each CATEGORY along with the REASONING:

{examples_text}

----------------------------------------------------

Now analyze the following new Input - Output pair.
Think step-by-step with the REASONING as shown in the examples, then provide your final classification.
The final line of your response must contain ONLY the CATEGORY letter (A, B, C, ...).
"""
COT_BASE_PROMPT = COT_BASE_PROMPT_TEMPLATE.format(examples_text=_generate_cot_examples_text(EXAMPLES))


# --- Strategy 3: Tree-of-Thought (ToT) ---
TOT_EXAMPLES_DATA = {
    "logical_consequence": [
        {
            "input": "Name: Hotel Torre Azul\nCity: El Arenal\nStar rating: 4",
            "output": "Hotel Torre Azul, a 4-star hotel, is located in El Arenal, Spain.",
            "explanation": "Output entails Input, because Output contains all the information in Input. Input does not entail Output, because Output contains some extra information (Spain)."
        },
        {
            "input": "Name: Hotel Torre Azul\nCity: El Arenal\nStar rating: 4\nHotel facilities: Lockers",
            "output": "Hotel Torre Azul, a 4-star hotel, is located in El Arenal.",
            "explanation": "Output does not entail Input, because Input contains some extra information (Hotel facilities: Lockers). Input entails Output, because Output does not contain any extra information."
        },
        {
            "input": "Name: Hotel Torre Azul\nCity: El Arenal\nRoom amenities: Balcony",
            "output": "Hotel Torre Azul is a charming hotel located in El Arenal.",
            "explanation": "Output does not entail Input, because Input contains some extra information (Room amenities: Balcony). Input does not entail Output, because Output contains some extra information (charming)."
        }
    ],
    "factually_wrong": [
        {
            "input": "Name: Hotel Torre Azul\nCity: El Arenal\nCountry: Spain\nStar rating: 4\nHotel facilities: Housekeeping\nRoom amenities: Balcony",
            "output": "Hotel Torre Azul is an incredibly charming 3-star hotel located in the center of El Arenal, Mexico.\nThe hotel facilities include free housekeeping and lockers. All rooms have a balcony.",
            "explanation": "The values 3, Mexico, and lockers are pieces of factually wrong information. The expressions incredibly charming (i), the center of (ii), All (iii), free (iv) are not pieces of factually wrong information."
        }
    ],
    "contradiction": [
        {
            "input": "Name: Hotel Torre Azul\nCity: El Arenal\nStar rating: 4",
            "output": "Hotel Torre Azul, a 3-star hotel, is located in El Arenal.",
            "explanation": "Input and Output contradict each other, since Input states that the hotel has 4 stars, while Output states that the hotel has 3 stars."
        }
    ],
    "self_contradiction": [
        {
            "input": "Name: Hotel Torre Azul\nCity: El Arenal\nStar rating: 4\nHotel facilities: Lockers",
            "output": "Hotel Torre Azul, a 4-star hotel, is located in El Arenal. The hotel facilities include lockers. The hotel does not provide lockers.",
            "explanation": "Output contradicts itself, since Output states that the hotel provides lockers and at the same time states that it does not provide them."
        }
    ]
}

TOT_BASE_PROMPT_TEMPLATE = """You are participating in an experiment.
You will see an Input that is a set of structured facts and an Output that is its natural language description.
Your task is to analyze the pair by answering a series of specific questions.

Here are definitions and examples for the concepts you will be asked about:

----------------------------------------------------

LOGICAL CONSEQUENCE:
Input entails Output if and only if by reading Input, you can infer that Output is true.
Output entails Input if and only if by reading Output, you can infer that Input is true.

Examples:

{logical_consequence_examples}

----------------------------------------------------

FACTUALLY WRONG INFORMATION:
When Input does not entail Output, Output contains information that is not in Input.
This information in Output may or may not be factually wrong.
We define as factually wrong any information in Output that conveys facts that could well turn out to be wrong, given the information in Input.
Factually wrong information does not comprise: (i) subjective opinions, (ii) vague or ambiguous statements, (iii) information that is unverifiable, (iv) information that can be inferred from Input to hold in most (but not all) cases.

Example:

{factually_wrong_examples}

----------------------------------------------------

CONTRADICTION:
Input and Output contradict each other if they contain information that cannot be true simultaneously.

Example:

{contradiction_examples}

----------------------------------------------------

SELF-CONTRADICTION:
Output contradicts itself if it contains pieces of information that cannot be true simultaneously.

Example:

{self_contradiction_examples}
"""
TOT_BASE_PROMPT = TOT_BASE_PROMPT_TEMPLATE.format(
    logical_consequence_examples="\n\n".join([f"Input:\n{ex['input']}\nOutput:\n{ex['output']}\nExplanation:\n{ex['explanation']}" for ex in TOT_EXAMPLES_DATA['logical_consequence']]),
    factually_wrong_examples="\n".join([f"Input:\n{ex['input']}\nOutput:\n{ex['output']}\nExplanation:\n{ex['explanation']}" for ex in TOT_EXAMPLES_DATA['factually_wrong']]),
    contradiction_examples="\n".join([f"Input:\n{ex['input']}\nOutput:\n{ex['output']}\nExplanation:\n{ex['explanation']}" for ex in TOT_EXAMPLES_DATA['contradiction']]),
    self_contradiction_examples="\n".join([f"Input:\n{ex['input']}\nOutput:\n{ex['output']}\nExplanation:\n{ex['explanation']}" for ex in TOT_EXAMPLES_DATA['self_contradiction']])
)


# ==================================
# 3. Helper Functions
# ==================================

def _call_openrouter(model: str, instruction: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A standardized helper to call the OpenRouter API."""
    api_key = None
    try:
        # Assumes the script is in a subdirectory (e.g., 'src') of the project root
        project_root = Path(__file__).resolve().parent.parent
        key_path = project_root / "api_key.txt"
        with open(key_path, "r") as f:
            api_key = f.read().strip()
    except (FileNotFoundError, IndexError):
        logging.error(f"API key file not found. Please create 'api_key.txt' in your project root directory: {project_root}")
        return None

    if not api_key:
        logging.error("API key file is empty.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert logician and evaluator of data-to-text/logic-to-text systems."},
            {"role": "user", "content": instruction}
        ],
        "temperature": config.get("temperature", 0.0)
    }

    try:
        # print(instruction)  # Debugging line to see the instruction being sent
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        # print(response.json())  # Debugging line to see the response structure
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling OpenRouter API: {e}")
        if e.response:
            logging.error(f"Response body: {e.response.text}")
        return None


def _extract_category(text: str) -> Optional[str]:
    """Extracts the first valid category letter from the model's response, searching backwards."""
    for char in reversed(text):
        if char in "ABCDEFGH":
            return char
    # Fallback: search forwards if not found
    for char in text:
        if char in "ABCDEFGH":
            return char
    logging.warning(f"Could not extract a valid category from response: '{text}'")
    return None


def _extract_yes_no(text: str) -> Optional[str]:
    """Extracts 'yes' or 'no' from the model's response, case-insensitive."""
    text_lower = text.lower()
    if "yes" in text_lower:
        return "yes"
    if "no" in text_lower:
        return "no"
    logging.warning(f"Could not extract 'yes' or 'no' from response: '{text}'")
    return None


# ==================================
# 4. Prompting Strategy Implementations
# ==================================

# --- Strategy 0: Binary Classification ---
def build_binary_classification_prompt(samples: List[Dict], new_input: str, new_output: str) -> str:
    prompt = BINARY_CLASSIFICATION_PROMPT
    for example in samples:
        input_str = "\n".join([f"{k}: {v}" for k, v in example['input'].items()])
        prompt += f"\n------------------------------\n\nInput:\n{input_str}\n\nOutput:\n{example['output']}\n\nANSWER: {example['category']}\n"
    prompt += f"\n------------------------------\n\nNow CLASSIFY the following:\n\nInput:\n{new_input}\n\nOutput:\n{new_output}\n\nANSWER (YES or NO):"
    return prompt


def run_binary_classification(model: str, item: Dict, config: Dict) -> Optional[str]:
    prompt = build_binary_classification_prompt(BINARY_EXAMPLES, item['input'], item['output'])
    response_data = _call_openrouter(model, prompt, {**config})
    if response_data and response_data.get("choices"):
        content = response_data["choices"][0]["message"]["content"]
        reasoning = response_data["choices"][0]["message"].get("reasoning", "")
        return _extract_yes_no(content), reasoning
    return None, None


# --- Strategy 0.1: 0-Shot ---
def build_zero_shot_prompt(new_input: str, new_output: str) -> str:
    prompt = ZERO_SHOT_BASE_PROMPT
    prompt += f"\n------------------------------\n\nNow CLASSIFY the following:\n\nInput:\n{new_input}\n\nOutput:\n{new_output}\n\nCATEGORY (write ONLY the letter):"
    return prompt


def run_zero_shot_classification(model: str, item: Dict, config: Dict) -> Optional[str]:
    prompt = build_zero_shot_prompt(item['input'], item['output'])
    response_data = _call_openrouter(model, prompt, {**config})
    if response_data and response_data.get("choices"):
        content = response_data["choices"][0]["message"]["content"]
        reasoning = response_data["choices"][0]["message"].get("reasoning", "")
        return _extract_category(content), reasoning
    return None, None


# --- Strategy 1: Few-Shot ---
def build_few_shot_prompt(samples: List[Dict], new_input: str, new_output: str) -> str:
    prompt = FEW_SHOT_BASE_PROMPT
    for sample in samples:
        input_str = "\n".join([f"{k}: {v}" for k, v in sample['input'].items()])
        prompt += f"\n------------------------------\n\nInput:\n{input_str}\n\nOutput:\n{sample['output']}\n\nCATEGORY: {sample['category']}\n"
    input_str = new_input
    prompt += f"\n------------------------------\n\nNow CLASSIFY the following:\n\nInput:\n{input_str}\n\nOutput:\n{new_output}\n\nCATEGORY (write ONLY the letter):"
    return prompt


def run_few_shot_classification(model: str, item: Dict, config: Dict) -> Optional[str]:
    prompt = build_few_shot_prompt(EXAMPLES, item['input'], item['output'])
    response_data = _call_openrouter(model, prompt, {**config})
    if response_data and response_data.get("choices"):
        content = response_data["choices"][0]["message"]["content"]
        reasoning = response_data["choices"][0]["message"].get("reasoning", "")
        return _extract_category(content), reasoning
    return None, None


# --- Strategy 2: Chain-of-Thought (CoT) ---
def build_cot_prompt(new_input: str, new_output: str) -> str:
    prompt = COT_BASE_PROMPT
    prompt += f"\nInput:\n{new_input}\n"
    prompt += f"\nOutput:\n{new_output}\n\nREASONING:\nCATEGORY (write ONLY the letter):"
    return prompt


def run_cot_classification(model: str, item: Dict, config: Dict) -> Optional[str]:
    prompt = build_cot_prompt(item['input'], item['output'])
    response_data = _call_openrouter(model, prompt, {**config})
    if response_data and response_data.get("choices"):
        content = response_data["choices"][0]["message"]["content"]
        reasoning = response_data["choices"][0]["message"].get("reasoning", "")
        return _extract_category(content), reasoning
    return None, None


# --- Strategy 3: Tree-of-Thought (ToT) ---
def build_tot_prompt(new_input: str, new_output: str, history: List[Tuple[str, str]]) -> str:
    prompt = TOT_BASE_PROMPT
    prompt += "\n----------------------------------------------------\n\n"
    prompt += "This is a new Input - Output pair. Please answer the question based on the definitions and examples above.\n\n"
    prompt += f"Input:\n{new_input}\n"
    prompt += f"\nOutput:\n{new_output}\n"
    if history:
        prompt += "\nYour previous judgments for this pair:\n"
        for q, ans, _ in history:
            prompt += f"- When asked '{q}', your answer was '{ans}'.\n"
    return prompt


def run_tot_decision(model: str, item: Dict, question: str, history: List, config: Dict) -> str:
    """Gets a single Yes/No decision from the model for a ToT step."""
    instruction = build_tot_prompt(item['input'], item['output'], history)
    instruction += f"\nBased on all the above, answer the following question. The final line of your response must contain ONLY YES or NO.\n"
    instruction += f"\nQuestion: {question}"

    response_data = _call_openrouter(model, instruction, {**config})
    if response_data and response_data.get("choices"):
        content = response_data["choices"][0]["message"]["content"]
        reasoning = response_data["choices"][0]["message"].get("reasoning", "")
        answer = _extract_yes_no(content)
        if answer:
            return answer, reasoning
        logging.warning(f"Could not extract a valid 'yes' or 'no' from response: '{content}'")
    logging.error(f"Invalid response from model: {response_data}")
    return None, None


def run_tot_classification(model: str, item: Dict, config: Dict) -> Tuple[Optional[List], Optional[str]]:
    """
    Executes the Tree-of-Thought classification, returning the decision history and the final category.
    Handles cases where answer is None by returning None for category and history.
    """
    history = []
    category = None

    answer, reasoning = run_tot_decision(model, item, "Does Output contradict itself?", history, config)
    history.append(("Does Output contradict itself?", answer, reasoning))
    if answer is None:
        return None, history
    if answer == "yes":
        category = "F"
    else:
        answer, reasoning = run_tot_decision(model, item, "Does Input entail Output?", history, config)
        history.append(("Does Input entail Output?", answer, reasoning))
        if answer is None:
            return None, history
        if answer == "yes":
            answer, reasoning = run_tot_decision(model, item, "Does Output entail Input?", history, config)
            history.append(("Does Output entail Input?", answer, reasoning))
            if answer is None:
                return None, history
            category = "A" if answer == "yes" else "C"
        else:
            answer, reasoning = run_tot_decision(model, item, "Does Output contain factually wrong information?", history, config)
            history.append(("Does Output contain factually wrong information?", answer, reasoning))
            if answer is None:
                return None, history
            if answer == "yes":
                answer, reasoning = run_tot_decision(model, item, "Does Output entail Input?", history, config)
                history.append(("Does Output entail Input?", answer, reasoning))
                if answer is None:
                    return None, history
                if answer == "yes":
                    category = "E"
                else:
                    answer, reasoning = run_tot_decision(model, item, "Do Input and Output contradict each other?", history, config)
                    history.append(("Do Input and Output contradict each other?", answer, reasoning))
                    if answer is None:
                        return None, history
                    category = "H" if answer == "yes" else "G"
            else:
                answer, reasoning = run_tot_decision(model, item, "Does Output entail Input?", history, config)
                history.append(("Does Output entail Input?", answer, reasoning))
                if answer is None:
                    return None, history
                category = "B" if answer == "yes" else "D"

    return category, history


# ==================================
# 5. Main Execution Logic
# ==================================

def load_data(data_path: Path) -> List[Dict]:
    """Loads and preprocesses data from the specified TSV file."""
    if not data_path.exists():
        logging.error(f"Data file not found at {data_path}")
        return []
    raw_data = pd.read_csv(data_path, sep="\t").dropna(subset=["block_Q"]).reset_index(drop=True)
    raw_data["item"] = raw_data["block_Q"].apply(lambda x: f"Item {int(x) - 4}")
    # Sort by 'block_Q' to ensure consistent order
    raw_data = raw_data.sort_values(by="block_Q").reset_index(drop=True)

    to_classify = raw_data.to_dict(orient='records')
    return to_classify


def run_experiments(config: Dict[str, Any]):
    """Main function to run classification experiments based on the configuration."""
    data_path = Path(config["data_path"])
    to_classify = load_data(data_path)
    if not to_classify:
        logging.info("No data to classify. Exiting.")
        return

    if "data_slice" in config:
        to_classify = to_classify[:config["data_slice"]]
        logging.info(f"Using a slice of the first {config['data_slice']} data items.")

    strategy_functions = {
        "binary": run_binary_classification,
        "zero_shot": run_zero_shot_classification,
        "few_shot": run_few_shot_classification,
        "cot": run_cot_classification,
        "tot": run_tot_classification,
    }
    strategy_func = strategy_functions.get(config["strategy"])
    if not strategy_func:
        raise ValueError(f"Unknown strategy: {config['strategy']}")

    for model_name in config["models"]:
        logging.info(f"--- Using model: {model_name} ---")
        model_results = []
        logging.info(f"--- Running classification with '{config['strategy']}' strategy ---")
        for item in tqdm(to_classify, desc=f"Classifying with {model_name}"):
            max_attempts = 5
            attempts = 0
            category, reasoning = strategy_func(model_name, item, config)
            while category is None and attempts < max_attempts - 1:
                attempts += 1
                logging.warning(f"No category returned for item. Retrying (attempt {attempts + 1}/{max_attempts})...")
                category, reasoning = strategy_func(model_name, item, config)
            model_results.append({
                "input": item['input'],
                "output": item['output'],
                "predicted_category": category,
                "reasoning": reasoning
            })

        # Save results for this model immediately
        results_dir = Path(config["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace("/", "_")
        output_filename = f"results_{config['strategy']}_{safe_name}.json"
        output_path = results_dir / output_filename
        logging.info(f"Saving results for {model_name} to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)


# ==================================
# 6. Entry Point
# ==================================

if __name__ == "__main__":
    # Define project root relative to the script's location
    project_root = Path(__file__).resolve().parent.parent
    
    # --- Main Configuration ---
    CONFIG = {
        "strategy": "zero_shot",  # Choose from: "binary", "zero_shot", "few_shot", "cot", "tot"
        "models": [
            "openai/gpt-5", # pass
            # "openai/o3", # harmless + courts
            "x-ai/grok-4", # pass
            "anthropic/claude-sonnet-4", # pass
            "anthropic/claude-opus-4.1", # pass
            "google/gemini-2.5-pro", # pass

            # "qwen/qwen3-235b-a22b-2507", # fail
            "deepseek/deepseek-r1-0528", # pass
            # "z-ai/glm-4.5-air", # fail + tech errors
            # "openai/gpt-oss-120b", # harmless + courts + error
            # "moonshotai/kimi-k2", # harmless + courts + error
            # "meta-llama/llama-4-maverick" # fail
        ],
        "data_path": project_root / "data" / "data_processed_hotel.tsv",
        "results_dir": project_root / "results" / "hotel",
        "temperature": 0.0,
        # "data_slice": 1, # Use a small slice for testing. Comment out to run on all data.
    }
    
    run_experiments(CONFIG)