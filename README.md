# Description
This repository contains the official code accompanying the [*SEM 2026](https://starsem2026.github.io/) paper _A Logic-Based Approach to Hallucinations in Data-to-Text NLG: Experiments with Human and LLM Annotators_.

Hallucinations are a persistent challenge in natural language generation, including data-to-text. [van Deemter (2024)](https://aclanthology.org/2024.cl-2.10/) introduced a framework based on the relation of logical consequence (“follows from”), which divides all data-to-text hallucinations into seven disjoint categories. We examine whether human annotators and large language models are able to apply the framework, in two data-to-text domains (hotel and logic). This work was carried out while the first author was interning at [trivago](https://www.trivago.com/).

The code is published mainly for reproducibility purposes.

# Setup
After cloning this repository, create a fresh conda environment:

```bash
conda create --name hallucination-framework --file requirements.txt python=3.13.4
```

# Structure

The repo is divided into three main subfolders, each of which has a similar internal structure.

`LLMs` contains the code (`src`) and data (`data`, including gold annotations, copied here from `Logic` and `Trivago` folders) used to perform annotation with LLMs in both domains, and the resulting annotations and code to analyze them (`results`).

`Logic` contains the data (`data`) used to perform annotation with humans in the logic domain, and the resulting annotations (crowdsourced in `evaluation/annotations` and gold in `evaluation/gold_label`) and code to analyze them (`evaluation`). For the logic domain, we leveraged a subset of data deriving from the [Grade Grinder Corpus](https://www.semanticscholar.org/paper/Student-Translations-of-Natural-Language-into-The-Barker-Plummer-Cox/28e805aae41255b8515173669ea19faa61e7cb87), preprocessed using the file in `evaluation/utils`.

`Trivago` contains the data (`data`) used to perform annotation with humans in the hotel domain, and the resulting annotations (crowdsourced in `experiment/crowdsourcing` (Prolific crowdworkers) and `experiment/trivago annotations` (trivago employees), and gold in `experiment/gold_label`), and code to analyze them (`experiment/analysis`). For the hotel domain, we leveraged data from the internal trivago database.

---

Note that:
- We cannot share the original Grade Grinder Corpus.
- We redacted part of the code, sensitive as belonging to trivago (e.g., data fetching and output generation for the hotel domain).
- Crowdworkers' information cannot be shared. We release the raw annotations anonymized, without IDs and demographic information.
- We redacted API keys. You need to acquire your own keys to run part of the code.
- You may need to adjust some paths.

# Citation

If you find this work helpful or use any artifact coming from it, please cite our paper as follows:

```bibtext
TBA
```

# License

The data and software are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
