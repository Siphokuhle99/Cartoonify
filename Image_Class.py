import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image, ImageTk
import sqlite3
import os

# Load the MobileNetV2 model
model = MobileNetV2(weights="imagenet", include_top=True)

# Database setup
DB_PATH = "feedback.db"

def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            prediction TEXT NOT NULL,
            feedback TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

initialize_database()

# Function to classify the image
def classify_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to 224x224 for the model
    img_array = np.array(img, dtype=np.float32)  # Convert to float32 for precision
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess with MobileNetV2 preprocessing

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    return decoded_predictions[:3]  # Return only the top 3 predictions

# Function to save feedback to the database
def save_feedback(image_path, prediction, feedback):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feedback (image_path, prediction, feedback) VALUES (?, ?, ?)",
                   (image_path, prediction, feedback))
    conn.commit()
    conn.close()

# Function to handle the upload button
def upload_image():
    global img_label, predictions_label, file_path, last_predictions
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        try:
            # Load and check image dimensions
            img = Image.open(file_path)
            if img.size[0] > 3000 or img.size[1] > 3000:
                messagebox.showerror("Error", "Please upload an image smaller than 3000x3000 pixels.")
                return

            # Display uploaded image
            img = img.resize((200, 200))  # Resize for display purposes
            img_tk = ImageTk.PhotoImage(img)
            img_label.config(image=img_tk)
            img_label.image = img_tk  # Keep a reference

            # Classify the image
            predictions = classify_image(file_path)

            # Format the result to include type, breed, and percentage
            result = ""
            for i, (imagenet_id, label, score) in enumerate(predictions):
                percentage = score * 100  # Convert to percentage
                result += f"{i+1}. {label} (Confidence: {percentage:.2f}%)\n"

            # Display predictions
            predictions_label.config(text=result.strip())

            # Store the predictions globally for feedback
            last_predictions = predictions

        except Exception as e:
            messagebox.showerror("Error", str(e))

# Function to submit feedback
def submit_feedback():
    feedback_text = feedback_entry.get("1.0", tk.END).strip()
    if not feedback_text:
        messagebox.showerror("Error", "Feedback cannot be empty.")
        return

    # Save feedback to database
    global file_path, last_predictions
    if last_predictions:
        save_feedback(file_path, str(last_predictions), feedback_text)
        messagebox.showinfo("Success", "Feedback submitted successfully!")
        feedback_entry.delete("1.0", tk.END)  # Clear the feedback box
    else:
        messagebox.showerror("Error", "No predictions to give feedback on.")

# Function to remove the uploaded image
def remove_image():
    img_label.config(image="")
    predictions_label.config(text="")
    feedback_entry.delete("1.0", tk.END)  # Clear feedback

# Create the GUI window
root = tk.Tk()
root.title("Image Classifier with Feedback")
root.geometry("800x700")
root.resizable(False, False)

# Background image (optional)
background_img = None
if os.path.exists("Images/Capture.JPG"):
    background_img = Image.open("Images/Capture.JPG")
    background_img = background_img.resize((800, 700))  # Resize to fit the window
    background_tk = ImageTk.PhotoImage(background_img)
    background_label = tk.Label(root, image=background_tk)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create and place the upload button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image, bg="lightblue", fg="black", font=("Arial", 14))
upload_btn.pack(pady=(20, 10))

# Create and place the remove button
remove_btn = tk.Button(root, text="Remove Image", command=remove_image, bg="lightblue", fg="black", font=("Arial", 14))
remove_btn.pack(pady=(0, 10))

# Label for displaying the uploaded image
img_label = tk.Label(root)
img_label.pack(pady=20)

# Label for displaying predictions
predictions_label = tk.Label(root, text="", bg="lightgrey", fg="black", font=("Arial", 12), wraplength=600)
predictions_label.pack(pady=(0, 20))

# Feedback entry box
feedback_frame = tk.Frame(root, bg="lightgrey")
feedback_frame.pack(pady=(10, 20))

feedback_label = tk.Label(feedback_frame, text="Provide Feedback:", bg="lightgrey", fg="black", font=("Arial", 12))
feedback_label.grid(row=0, column=0, padx=10)

feedback_entry = tk.Text(feedback_frame, width=50, height=5, font=("Arial", 12))
feedback_entry.grid(row=1, column=0, padx=10, pady=5)

# Submit feedback button
feedback_btn = tk.Button(root, text="Submit Feedback", command=submit_feedback, bg="green", fg="white", font=("Arial", 14))
feedback_btn.pack(pady=(0, 20))

# Start the GUI main loop
root.mainloop()
