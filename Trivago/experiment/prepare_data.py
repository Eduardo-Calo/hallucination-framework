import ast
import pandas as pd
import re

def transform_input_string(input_str):
    def transform_data(data_str):
        data_dict = ast.literal_eval(data_str)
        transformed_data = []

        for key, value in data_dict.items():
            # Replace underscores with spaces in keys
            formatted_key = key.replace('_', ' ')

            if isinstance(value, int): value = str(value)

            if not value or value == ['']:
                transformed_data.append(f'{formatted_key.capitalize()}: None') # outdated
            elif len(value) == 1:
                transformed_data.append(f'{formatted_key.capitalize()}: {value[0]}') # for rating
            elif isinstance(value, str) and len(value) > 1:
                transformed_data.append(f'{formatted_key.capitalize()}: {value}') # for names etc
            else:
                transformed_data.append(f'{formatted_key.capitalize()}: {", ".join(value)}')

        return '\n'.join(transformed_data)

    # Split the input string into separate dictionaries
    data_list = [item.strip() for item in input_str.split('}') if item.strip()]

    # Apply the transformation to each dictionary string in the list
    transformed_data_list = [transform_data(item + '}') for item in data_list]

    return '\n'.join(transformed_data_list)

def format_input_string(input_str):
    # Split the input string into sentences
    sentences = input_str.split('.')

    # Define a regular expression pattern for replacing commas
    comma_pattern = re.compile(r'\b,\b')

    # Format each sentence
    formatted_sentences = []
    for sentence in sentences:
        # Remove leading and trailing whitespaces
        sentence = sentence.strip()
        if sentence:
            # Add a period at the end of each sentence
            sentence += '.'
            # Add spaces after commas
            sentence = re.sub(comma_pattern, ', ', sentence)
            formatted_sentences.append(sentence)

    # Join the formatted sentences into a single string
    formatted_string = '\n'.join(formatted_sentences)
    return formatted_string


df = pd.read_csv('../data/data.csv', sep=',', converters={'input': transform_input_string, 'output': format_input_string})
df.to_csv('../data/data_processed.tsv', sep='\t', index=False)