# Balochi GPT Text Generation

Fine-tunes `goldfish-models/bgp_latn_full` on the `Balochi_sentences` column of `Aynkader/Balochi-dataset`.

## Important data note
The source corpus is a parallel sentence corpus, not a proverb collection. The notebook extracts only Balochi text. Add cultural proverbs only when collected from permitted sources and verified by fluent speakers. Do not label ordinary sentences as proverbs.

## Run locally
```bash
pip install -r requirements.txt
# Set HF_MODEL_ID to your uploaded fine-tuned model repository
streamlit run app.py
```

Generated text needs human linguistic and cultural review.
