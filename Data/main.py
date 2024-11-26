import tkinter as tk
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import subprocess
import pkg_resources

def check_and_install_dependencies():
    """Check and install required packages"""
    required_packages = {
        'torch',
        'transformers',
        'sentencepiece',
        'pandas',
        'scikit-learn'
    }
    
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = required_packages - installed_packages
    
    if missing_packages:
        print("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
            # Restart the script after installing packages
            os.execv(sys.executable, ['python'] + sys.argv)
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {str(e)}")
            sys.exit(1)

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"medical_chatbot_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_required_files():
    required_files = [
        'Training.csv',
        'Testing.csv',
        'symptom_Description.csv',
        'symptom_precaution.csv',
        'Symptom_severity.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def check_model_files():
    model_dir = Path("medical_model")
    return model_dir.exists() and (model_dir / "pytorch_model.bin").exists()

def initialize_system(logger):
    try:
        # Check and install dependencies first
        check_and_install_dependencies()
        
        # Check required files
        missing_files = check_required_files()
        if missing_files:
            logger.error(f"Missing required files: {', '.join(missing_files)}")
            return False
        
        # Check model files
        if not check_model_files():
            logger.warning("Pre-trained model not found. Training new model...")
            from train import main as train_model
            train_model()
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return False

def create_gui(logger):
    try:
        root = tk.Tk()
        root.title("Medical Assistant Chatbot")
        
        window_width = 800
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        return root
        
    except Exception as e:
        logger.error(f"Error creating GUI: {str(e)}")
        return None

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Medical Chatbot System")
    
    try:
        # Initialize system
        if not initialize_system(logger):
            logger.error("System initialization failed")
            return
        
        # Create GUI
        root = create_gui(logger)
        if not root:
            logger.error("GUI creation failed")
            return
        
        # Create chatbot application
        from chatbot_gui import ChatbotGUI
        app = ChatbotGUI(root)
        logger.info("Medical Chatbot GUI initialized successfully")
        
        # Start the application
        root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, logger))
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        sys.exit(1)

def on_closing(root, logger):
    try:
        logger.info("Shutting down Medical Chatbot System")
        root.destroy()
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 