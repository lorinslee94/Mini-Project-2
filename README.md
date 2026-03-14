# Mini Project 2: Part 3 and Part 4

This repository contains the implementation and reports for **Part 3 (Multi-Agent Chatbot)** and **Part 4 (LLM Evaluation)**.

## File Structure

```
.
├── app.py
├── part3/
├── part4/
├── part3_report_lorin_lee.pdf
├── part4_report_lorin_lee.pdf
├── README.md
└── requirements.txt
```

## Root Directory

- `app.py` — Streamlit entry point for the chatbot UI.  
  It imports and uses modules from `part3/`.

- `part3_report_lorin_lee.pdf` — Final report for Part 3.  
- `part4_report_lorin_lee.pdf` — Final report for Part 4.  
- `requirements.txt` — Project dependencies.

## part3/

Contains all Python source code for the **Multi-Agent Chatbot** (core logic, agents, routing, retrieval, utilities).  
The Streamlit UI is implemented in `app.py`.

## part4/

Contains the implementation for **LLM Evaluation**, including evaluation scripts and prompt generation.

- `test_set.json` — prompts used for evaluation

## Setup Instructions

### 1. Create Conda Environment

```bash
conda create -n multi_agent_env python=3.11
conda activate multi_agent_env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Set the following environment variables:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`

Example (macOS/Linux):

```bash
export OPENAI_API_KEY="your_key_here"
export PINECONE_API_KEY="your_key_here"
```

> Note:
> - The Pinecone index name is hardcoded in the source code as:  
>   `ml-textbook-rag-1536`
> - `app.py` is located in the root directory and must be run from the root of the project.

## Running Part 3 – Multi-Agent Chatbot

The Streamlit application must be executed from the **root directory** of the project.

### Start the Application

```bash
streamlit run app.py
```

This launches the Multi-Agent Chatbot interface in your browser.

- `app.py` is located in the root directory.
- The core chatbot logic is implemented inside the `part3/` module.
- The Streamlit UI imports and uses the `Head_Agent` and supporting agents from `part3/`.

Once running, you can interact with the chatbot directly through the web interface.

## Running Part 4 Evaluation

All commands below must be executed from the **root directory** of the project.

### 1. Generate Evaluation Prompts (Optional)

```bash
python -m part4.dataset_generator
```

This will generate `test_set.json` inside the `part4/` directory.

> Note:
> - A pre-generated `test_set.json` is included with the submission.
> - If you run the generator again, the original file will be **overwritten**.

### 2. Run Evaluation

After the dataset has been generated (or using the provided file), run:

```bash
python -m part4.run_evaluation
```

Once evaluation completes, the results will be saved as:

```
part4/eval_results.json
```

This file contains the structured evaluation metrics and LLM-judge outputs.


