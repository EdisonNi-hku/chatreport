## Zip files
- chatgpt_annotation.zip contains the annotations on ChatGPT (June version) outputs about hallucinations on content/source dimensions.
- gpt-4_annotation.zip contains the annotations on GPT-4 (June version) outputs about hallucinations on content/source dimensions.
- sustainability_reports.zip contains the 10 sustainability reports that the data based on.

## Structure of the JSON files:
```
{
  "Report Names (e.g., NYSE_AIZ_2022)": {
    "Question IDs (e.g., tcfd_1)": {
      "question": "The TCFD question we asked about the sustainability report",
      "answer": "The LLM’s answer to the question",
      "sources": ["a list of source numbers that the LLM cited to answer the question"],
      "relevant_chunks (i.e., evidence cited by the LLM)": {
        "source_number_#1": "source #1 content",
        "source_number_#2": "source #2 content",
        ...
      },
      "irrelevant_chunks (i.e., evidence not cited by the LLM)": {
        "source_number_#1": "source #1 content",
        "source_number_#2": "source #2 content",
        ...
      }
    },
    "tcfd_2": ...,
    "tcfd_3": ...,
    ...
    "tcfd_11": ...,      
  }
  "NYSE_BV_2022": ...
  ...
}
```