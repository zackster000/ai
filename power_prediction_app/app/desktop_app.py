import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path so we can import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the predictor, if it fails, catch it later
try:
    from model.predict import Predictor
except ImportError:
    Predictor = None

class DesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Electric Consumption Predictor")
        self.root.geometry("450x550")
        self.root.configure(bg="#f0f2f6")
        
        # Style
        self.style = ttk.Style()
        self.style.configure("TLabel", background="#f0f2f6", font=("Helvetica", 10))
        self.style.configure("TButton", font=("Helvetica", 10, "bold"))
        self.style.configure("Header.TLabel", font=("Helvetica", 14, "bold"), foreground="#2c3e50")
        
        # UI Elements
        self.create_widgets()
        
        # Load Predictor
        try:
            self.predictor = Predictor(model_path='model/saved_model.pkl')
            self.status_label.config(text=f"Model Loaded: {self.predictor.model_name}", foreground="green")
        except Exception as e:
            self.predictor = None
            self.status_label.config(text="Model NOT found! Run training first.", foreground="red")
            messagebox.showwarning("Warning", f"Could not load the model: {e}")

    def create_widgets(self):
        # Header
        header = ttk.Label(self.root, text="Electric Consumption Predictor", style="Header.TLabel")
        header.pack(pady=20)
        
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="20", style="Card.TFrame")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Input Section
        ttk.Label(main_frame, text="Prediction Date (YYYY-MM-DD):").pack(anchor="w", pady=(10, 0))
        self.date_entry = ttk.Entry(main_frame)
        self.date_entry.insert(0, (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"))
        self.date_entry.pack(fill="x", pady=5)
        
        ttk.Label(main_frame, text="Last Consumption (kWh):").pack(anchor="w", pady=(10, 0))
        self.last_cons_entry = ttk.Entry(main_frame)
        self.last_cons_entry.insert(0, "25.0")
        self.last_cons_entry.pack(fill="x", pady=5)
        
        # Predict Button
        self.predict_btn = ttk.Button(main_frame, text="🚀 Predict", command=self.predict)
        self.predict_btn.pack(pady=30)
        
        # Output Section
        self.result_label = ttk.Label(main_frame, text="", font=("Helvetica", 12, "bold"))
        self.result_label.pack(pady=10)
        
        self.details_label = ttk.Label(main_frame, text="", font=("Helvetica", 9))
        self.details_label.pack(pady=5)
        
        # Status Bar
        self.status_label = ttk.Label(self.root, text="Status: Ready", font=("Helvetica", 8))
        self.status_label.pack(side="bottom", fill="x", padx=10, pady=5)

    def predict(self):
        if not self.predictor:
            messagebox.showerror("Error", "No trained model found! Please run train_model.py first.")
            return
        
        target_date = self.date_entry.get()
        try:
            last_val = float(self.last_cons_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Last consumption must be a valid number.")
            return
            
        try:
            # Simple date validation
            datetime.strptime(target_date, "%Y-%m-%d")
            
            # Predict
            prediction = self.predictor.predict_single(target_date, last_val)
            
            # Show result
            self.result_label.config(text=f"Predicted Value: {prediction} kWh", foreground="#e67e22")
            self.details_label.config(text=f"Forecast for: {target_date}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DesktopApp(root)
    root.mainloop()
