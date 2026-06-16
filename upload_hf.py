from huggingface_hub import HfApi
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load local model
tokenizer = DistilBertTokenizer.from_pretrained('models/distilbert_branchC')
model = DistilBertForSequenceClassification.from_pretrained('models/distilbert_branchC')

# Push to HuggingFace
tokenizer.push_to_hub('bishnoiavantika1/phqure-distilbert-branchc')
model.push_to_hub('bishnoiavantika1/phqure-distilbert-branchc')

print("Uploaded to HuggingFace successfully.")