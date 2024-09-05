import random
import torch
import tkinter as tk
import customtkinter as ctk
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import json

# Load your model and data
with open('intents_clg.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "N33V"

# Create a custom Tkinter window
root = ctk.CTk()
root._set_appearance_mode("dark")  # Set initial theme

root.title("Jain Enquiry Bot")
root.geometry("400x600")

# Create a Canvas widget for the background
bg_canvas = ctk.CTkCanvas(root)
bg_canvas.pack(fill=tk.BOTH, expand=True)
bg_canvas.create_rectangle(0, 0, root.winfo_screenwidth(), root.winfo_screenheight(), fill="#E0E0E0", outline="")

# Create a Text widget to display the conversation
chat_text = ctk.CTkTextbox(bg_canvas, wrap=tk.WORD, font=("Helvetica", 20), fg_color="black")  # Font size and color adjustments
chat_text.pack(fill=tk.BOTH, expand=True)
chat_text.configure(state=tk.DISABLED)

# Create an Entry widget for user input
input_field = ctk.CTkEntry(root, font=("Helvetica", 20), fg_color="black")  # Font size and color adjustments
input_field.pack(fill=tk.X, padx=10, pady=10)

def send_message():
    user_input = input_field.get()
    display_message(f"You: {user_input}", is_user=True)

    # Process user input and generate bot response (similar to your code)
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
                display_message(f"{bot_name}: {bot_response}")
    else:
        display_message(f"{bot_name}: I do not understand...", is_user=True)
    
    input_field.delete(0, tk.END)  # Clear the input field

def display_message(message, is_user=False):
    tag = "user" if is_user else "bot"

    chat_text.configure(state=tk.NORMAL)
    chat_text.insert(tk.END, message + "\n", tag)
    chat_text.tag_config(tag, background="black" if is_user else "#2c3e50")
    chat_text.see(tk.END)
    chat_text.configure(state=tk.DISABLED)

send_button = ctk.CTkButton(root, text="Send", command=send_message, font=("Helvetica", 16))
send_button.pack()

input_field.bind("<Return>", lambda event: send_message())


def toggle_theme():
    current_theme = root._get_appearance_mode()
    new_theme = "light" if current_theme == "dark" else "dark"
    root._set_appearance_mode(new_theme)

theme_button = ctk.CTkButton(root, text="Toggle Theme", command=toggle_theme, font=("Helvetica", 16))
theme_button.pack()

root.mainloop()


