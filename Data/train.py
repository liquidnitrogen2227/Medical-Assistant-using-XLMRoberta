import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AdamW
from model import MedicalDataset

def train_model(model, train_loader, device, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def main():
    # Load and prepare data
    df = pd.read_csv('Training.csv')
    
    # Convert symptoms to text format
    symptom_columns = df.columns[:-1]
    texts = []
    for _, row in df.iterrows():
        symptoms = [col for col in symptom_columns if row[col] == 1]
        texts.append(" ".join(symptoms))
    
    # Prepare labels
    labels = pd.factorize(df.iloc[:, -1])[0]
    
    # Initialize tokenizer and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=len(df.iloc[:, -1].unique())
    ).to(device)
    
    # Create dataset and dataloader
    dataset = MedicalDataset(texts, labels, tokenizer)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Train the model
    train_model(model, train_loader, device)
    
    # Save the model
    model.save_pretrained('medical_model')
    tokenizer.save_pretrained('medical_model')

if __name__ == "__main__":
    main() 