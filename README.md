# chatreport
Github implementation of CHATREPORT. This is a snapshot of the v1.0 backend of our [web application](https://reports.chatclimate.ai/) on 2023.07.28.
We release anonymized user questions and ChatReport answers [here](https://drive.google.com/file/d/1rek6quK6YnxQx5JlRMv91KhMznt-ZFPJ/view?usp=sharing). The cut-off point of user QA is on 11.09.2023.

## Directories
- annotated_data: annotation results of our human evaluation
  - chatgpt/gpt-4_results.json: ChatGPT/GPT-4's outputs on sampled reports.
  - chatgpt/gpt-4_annotation_zip: our annotations on ChatGPT/GPT-4's outputs
- generated_data
  - analysis_reports.zip: a collection of 1015 analysis reports generated by CHATREPORT about NYSE companies.
- code
  - auto_prompt_engineer.py: the script for transferring experts' feedback to prompt engineering.
  - app.py: the script handling application run
  - document.py: the script handling PDF reading and retrieval
  - reader.py: TCFD questions & conformity analyses
  - user_qa.py: the script handling customized analyses through question answering

## Usage
Set up the environment
```shell
conda create --name chatreport python=3.10
conda activate chatreport
pip install -r requirements.txt
```

1. Modify `OPENAI_API_KEYS` in `apikey.ini`, you could fill in multiple api key like below
```python
OPENAI_API_KEYS = ['sk-XXX', 'sk-XXX']
```

2. Analyze a given report, for example: NYSE_SNE_2018.pdf
```commandline
python app.py --pdf_path NYSE_SNE_2018.pdf
```
- Analysis report will be stored at "NYSE_SNE_2018.html"
- Answers to TCFD bullet points will be stored at "data/answers/NYSE_SNE_2018.json"
- Assessments w.r.t. TCFD guidelines will be stored at "data/assessment/NYSE_SNE_2018.json"
- Basic info will be stored at "data/basic_info/NYSE_SNE_2018.json"
- The original PDF will be stored at "data/pdf/NYSE_SNE_2018.pdf"
- Embedding/vector database will be stored at "data/vector_db/NYSE_SNE_2018/". When the report is queried again, there's no need to re-embed the vector DB.

3. Conduct customized Question Answering
```shell
python app.py --pdf_path NYSE_SNE_2018.pdf --user_question "What is the level of cheap talk in the report?" --answer_length 50
```
- user_question takes the user's question
- answer_length specifies the length of generation (we recommend 50)
- Questions, answers, sources, pages, and answer length will be appended to "data/user_qa/NYSE_SNE_2018.jsonl" (lines of JSONs)

## Citation
Please cite our paper if you use CHATREPORT in your research.
```bibtex
@misc{ni2023chatreport,
      title={CHATREPORT: Democratizing Sustainability Disclosure Analysis through LLM-based Tools}, 
      author={Jingwei Ni and Julia Bingler and Chiara Colesanti-Senni and Mathias Kraus and Glen Gostlow and Tobias Schimanski and Dominik Stammbach and Saeid Ashraf Vaghefi and Qian Wang and Nicolas Webersinke and Tobias Wekhof and Tingyu Yu and Markus Leippold},
      year={2023},
      eprint={2307.15770},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

