import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#setup and show device being used (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_df = pd.read_csv("data/clean/train.csv")
train_df = train_df.head(5000)
val_df = pd.read_csv("data/clean/validation.csv")

#tokenization (BERT specific)
#we use 'bert-base-uncased' because it handles lowercase and subwords efficently
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

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
        
        #return as a dictionary for the model
        encoding = self.tokenizer(headline, padding='max_length', max_length=32, truncation=True, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label
        }

#dataLoaders
train_dataset = ClickbaitDataset(train_df["headline"].tolist(), train_df["label"].tolist(), tokenizer)
val_dataset = ClickbaitDataset(val_df["headline"].tolist(), val_df["label"].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

#model BERT + a Linear Classification Layer
class ClickbaitBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]   # DistilBERT has no pooler_output
        return self.classifier(cls)

model = ClickbaitBertModel().to(device)

#training
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3  #BERT usually needs fewer epochs
print("Starting the training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
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
        
        if i % 10 == 0:
            print(f"  batch {i}/{len(train_loader)} - loss {loss.item():.4f}", flush=True)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

#results
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

print("\n BERT Model Evaluation ")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds))
print("Recall:", recall_score(all_labels, all_preds))
print("F1 Score:", f1_score(all_labels, all_preds))

#save the model for easy loading later
torch.save(model.state_dict(), "bert_clickbait_model.pt")
print("\nModel saved as bert_clickbait_model.pt")

#quick test without retraining
def get_score(headline):
    model.eval()
    inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=32 )
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

