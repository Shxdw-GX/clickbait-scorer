import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Setup and Data Loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_df = pd.read_csv("data/clean/train.csv")
val_df = pd.read_csv("data/clean/validation.csv")

# 2. Tokenization (BERT specific)
# We use 'bert-base-uncased' because it handles lowercase and subwords beautifully
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ClickbaitDataset(Dataset):
    def __init__(self, headlines, labels, tokenizer):
        self.headlines = headlines
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, item):
        headline = str(self.headlines[item])
        label = torch.tensor(self.labels[item], dtype=torch.long)
        
        # Tokenize and return as a dictionary for the model
        encoding = self.tokenizer(headline, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label
        }

# Prepare DataLoaders
train_dataset = ClickbaitDataset(train_df["headline"].tolist(), train_df["label"].tolist(), tokenizer)
val_dataset = ClickbaitDataset(val_df["headline"].tolist(), val_df["label"].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 3. Model Definition (BERT + a Linear Classification Layer)
class ClickbaitBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # We take the [CLS] token output (the first token of every sequence) 
        # and pass it through a linear layer to get our binary classification.
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output represents the [CLS] token's meaning for the whole sentence
        pooled_output = outputs.pooler_output 
        logits = self.classifier(pooled_output)
        return logits

model = ClickbaitBertModel().to(device)

# 4. Training Setup
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3  # BERT usually needs fewer epochs than Logistic Regression to converge
print("Starting training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

# 5. Final Evaluation (Person 4's "Stronger Model" results)
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1).flatten()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n--- BERT Model Evaluation ---")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds))
print("Recall:", recall_score(all_labels, all_preds))
print("F1 Score:", f1_score(all_labels, all_preds))

# 6. Save the Model for Person 5's Demo
torch.save(model.state_dict(), "bert_clickbait_model.pt")
print("\nModel saved as bert_clickbait_model.pt")

# Quick Test for the team:
def get_score(headline):
    model.eval()
    inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
        # Get probability of class 1 (clickbait)
        prob = torch.softmax(outputs, dim=1)[0][1].item()
    return prob

test_headlines = [
    "You Won't Believe What This Dog Did Next",
    "Senate passes budget bill after late night vote",
    "17 Things Only 90s Kids Will Understand",
]

for h in test_headlines:
    print(f"{round(get_score(h), 3)} | {h}")

