import pandas as pd
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader

class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MedicalChatbot:
    def __init__(self):
        try:
            # Load datasets first
            self.load_datasets()
            
            # Initialize model and tokenizer
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_name = 'xlm-roberta-base'
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name)
            self.model = XLMRobertaForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.get_unique_conditions())
            ).to(self.device)
            
            # Initialize symptoms list
            self.current_symptoms = []
            
        except Exception as e:
            print(f"Error initializing Medical Chatbot: {str(e)}")
            raise

    def load_datasets(self):
        """Load all required datasets"""
        try:
            self.train_data = pd.read_csv('Training.csv')
            self.description_data = pd.read_csv('symptom_Description.csv')
            self.precaution_data = pd.read_csv('symptom_precaution.csv')
            self.severity_data = pd.read_csv('Symptom_severity.csv')
        except FileNotFoundError as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def is_medical_related(self, message):
        """Check if the message contains medical-related terms"""
        medical_terms = [
            'pain', 'ache', 'fever', 'cough', 'headache', 'nausea', 
            'dizzy', 'tired', 'sore', 'sick', 'hurt', 'doctor', 
            'medicine', 'symptom', 'treatment', 'hospital', 'disease',
            'infection', 'illness', 'condition'
        ]
        return any(term in message.lower() for term in medical_terms)

    def get_unique_conditions(self):
        """Get list of unique medical conditions"""
        return self.train_data.iloc[:, -1].unique()

    def get_condition_description(self, condition):
        """Get description for a specific condition"""
        try:
            return self.description_data[
                self.description_data['disease'] == condition
            ]['description'].iloc[0]
        except (KeyError, IndexError):
            return "Description not available"

    def get_condition_precautions(self, condition):
        """Get precautions for a specific condition"""
        try:
            precautions = self.precaution_data[
                self.precaution_data['Disease'] == condition
            ].iloc[0, 1:].tolist()
            return [p for p in precautions if isinstance(p, str) and p.strip()]
        except (KeyError, IndexError):
            return []

    def predict_condition(self, symptoms):
        """Predict medical condition based on symptoms"""
        if not symptoms:
            return None
            
        # Convert symptoms to text format
        symptom_text = " ".join(symptoms)
        
        # Tokenize the input
        inputs = self.tokenizer(
            symptom_text,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_idx = torch.argmax(predictions, dim=1).item()
            
        conditions = self.get_unique_conditions()
        return conditions[predicted_idx]

    def add_symptom(self, symptom):
        """Add a symptom to the current list"""
        if symptom not in self.current_symptoms:
            self.current_symptoms.append(symptom)
        return self.current_symptoms

    def clear_symptoms(self):
        """Clear the current symptoms list"""
        self.current_symptoms = []

    def get_current_symptoms(self):
        """Get the current list of symptoms"""
        return self.current_symptoms
