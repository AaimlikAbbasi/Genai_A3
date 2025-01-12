import tkinter as tk
from tkinter import scrolledtext
import torch
from q1.q1 import TransformerModel  # Import your Transformer model class
import sentencepiece as spm

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(src_vocab=3000, tgt_vocab=3000).to(device)
model.load_state_dict(torch.load("models/q1/checkpoints/transformer_epoch_10.pth", map_location=device))
model.eval()

# Load SentencePiece tokenizers
sp_en = spm.SentencePieceProcessor(model_file="models/q1/spm_en.model")
sp_ur = spm.SentencePieceProcessor(model_file="models/q1/spm_ur.model")

# Translation Function
def translate(text):
    src_ids = sp_en.encode(text, add_bos=True, add_eos=True)
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    tgt_input = torch.tensor([sp_ur.bos_id()]).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(50):
            output = model(src_tensor, tgt_input)
            next_token = output.argmax(-1)[:, -1].item()
            tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]]).to(device)], dim=1)
            if next_token == sp_ur.eos_id():
                break

    tgt_tokens = tgt_input.squeeze().cpu().tolist()
    return sp_ur.decode(tgt_tokens[1:-1])

# GUI Setup
root = tk.Tk()
root.title("English-to-Urdu Translator")

# Input Textbox
tk.Label(root, text="Enter English Text:").pack(pady=5)
english_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=5)
english_input.pack(pady=5)

# Translate Button
def on_translate():
    english_text = english_input.get("1.0", tk.END).strip()
    urdu_translation = translate(english_text)
    urdu_output.config(state='normal')
    urdu_output.delete("1.0", tk.END)
    urdu_output.insert(tk.END, urdu_translation)
    urdu_output.config(state='disabled')

tk.Button(root, text="Translate", command=on_translate).pack(pady=5)

# Output Textbox
tk.Label(root, text="Urdu Translation:").pack(pady=5)
urdu_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=5, state='disabled')
urdu_output.pack(pady=5)

# Run the App
root.mainloop()
