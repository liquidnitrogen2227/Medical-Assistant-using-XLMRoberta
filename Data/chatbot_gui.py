import tkinter as tk
from tkinter import ttk, scrolledtext
from model import MedicalChatbot
from datetime import datetime

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Chatbot Assistant")
        
        # Initialize chatbot
        self.chatbot = MedicalChatbot()
        
        # Configure main window
        self.setup_gui()
        
        # Initialize state variables
        self.waiting_for_symptoms = False
        self.current_condition = None
        
        # Start conversation
        self.display_message("Bot: Hello! I'm your medical assistant. How can I help you today?")

    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create chat display
        self.chat_display = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=50, height=20
        )
        self.chat_display.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Create symptoms label
        self.symptoms_label = ttk.Label(
            main_frame, text="Current Symptoms: None", wraplength=300
        )
        self.symptoms_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Create input field
        self.input_field = ttk.Entry(main_frame, width=40)
        self.input_field.grid(row=2, column=0, pady=5)
        self.input_field.bind("<Return>", lambda e: self.send_message())
        
        # Create send button
        send_button = ttk.Button(
            main_frame, text="Send", command=self.send_message
        )
        send_button.grid(row=2, column=1, pady=5, padx=5)

    def display_message(self, message):
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_display.insert(tk.END, f"[{timestamp}] {message}\n")
        self.chat_display.see(tk.END)

    def send_message(self):
        message = self.input_field.get().strip()
        if message:
            self.display_message(f"You: {message}")
            self.input_field.delete(0, tk.END)
            self.process_message(message)

    def process_message(self, message):
        if not self.chatbot.is_medical_related(message):
            self.display_message("Bot: I'm a medical assistant. Please tell me about your medical symptoms or concerns.")
            return

        if "help" in message.lower():
            self.display_message("Bot: I can help you identify possible medical conditions based on your symptoms. Please describe your symptoms one by one.")
            self.waiting_for_symptoms = True
            return

        if self.waiting_for_symptoms:
            if self.is_no_more_symptoms(message):
                self.predict_condition()
                self.waiting_for_symptoms = False
            else:
                cleaned_message = self.clean_message(message)
                self.chatbot.add_symptom(cleaned_message)
                symptoms = self.chatbot.get_current_symptoms()
                self.update_symptoms_label(symptoms)
                self.display_message("Bot: Any other symptoms? (Say 'no' if done)")

    def is_no_more_symptoms(self, message):
        no_more = ['no', 'none', 'nothing', 'that\'s all', 'thats all', 'done']
        return any(phrase in message.lower() for phrase in no_more)

    def clean_message(self, message):
        phrases_to_remove = [
            'i am', 'i have', 'i\'m', 'suffering from', 'experiencing',
            'feeling', 'with', 'got', 'having'
        ]
        message = message.lower()
        for phrase in phrases_to_remove:
            message = message.replace(phrase, '')
        return message.strip()

    def predict_condition(self):
        symptoms = self.chatbot.get_current_symptoms()
        if not symptoms:
            self.display_message("Bot: No symptoms provided. Please tell me your symptoms.")
            return

        condition = self.chatbot.predict_condition(symptoms)
        if condition:
            self.current_condition = condition
            self.display_message(f"Bot: Based on your symptoms, you might have {condition}.")
            
            # Get and display condition description
            description = self.chatbot.get_condition_description(condition)
            self.display_message(f"Bot: {description}")
            
            # Get and display precautions
            precautions = self.chatbot.get_condition_precautions(condition)
            if precautions:
                self.display_message("Bot: Recommended precautions:")
                for i, precaution in enumerate(precautions, 1):
                    self.display_message(f"{i}. {precaution}")
            
            self.display_message("Bot: Please consult a healthcare professional for proper diagnosis and treatment.")
            self.chatbot.clear_symptoms()
            self.update_symptoms_label([])
        else:
            self.display_message("Bot: I couldn't determine the condition. Please consult a healthcare professional.")

    def update_symptoms_label(self, symptoms):
        if symptoms:
            symptoms_text = "Current Symptoms: " + ", ".join(symptoms)
        else:
            symptoms_text = "Current Symptoms: None"
        self.symptoms_label.config(text=symptoms_text)

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 