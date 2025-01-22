import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from transformers import GPT2Tokenizer

# Load GPT2 tokenizer
def load_gpt2_tokenizer():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return tokenizer

# Tokenize text using GPT2 tokenizer
def tokenize_text(input_tensor, tokenizer):
    return input_tensor  # Input tensor is already tokenized

# Save tensor slices to .npz files
def save_tensor_slice(x, y, index, output_dir):
    try:
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()

        file_name = f"tensor_{index}.npz"
        file_path = os.path.join(output_dir, file_name)

        np.savez(file_path, x=x_np, y=y_np)
        print(f"Saved tensor_{index}.npz to {file_path}")
    except Exception as e:
        print(f"Error saving tensor_{index}: {e}")

# Main function
def main():
    TENSOR_FILE = "/media/sien/media/code/implementation_llms_ascard/encoded_context.pt"
    output_dir = "/media/sien/media/data/train_data/"
    os.makedirs(output_dir, exist_ok=True)

    try:
        input_data = torch.load(TENSOR_FILE)
    except Exception as e:
        print(f"Error loading tensor file: {e}")
        return

    if not isinstance(input_data, dict) or 'input_ids' not in input_data:
        print("Loaded data is not in the expected format or missing 'input_ids'.")
        return

    input_tensor = input_data['input_ids']

    if input_tensor.ndimension() < 2:
        input_tensor = input_tensor.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_gpt2_tokenizer()

    batch_size = input_tensor.size(0)
    seq_len = input_tensor.size(1)
    num_slices = seq_len // 1024

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for idx in range(num_slices):
            start_idx = idx * 1024
            end_idx = (idx + 1) * 1024
            slice_tensor = input_tensor[:, start_idx:end_idx]
            x = slice_tensor
            y = torch.roll(x, shifts=-1, dims=1)
            y[:, -1] = 0
            tokenized_x = tokenize_text(x, tokenizer)
            futures.append(executor.submit(save_tensor_slice, tokenized_x, y, idx, output_dir))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in future: {e}")

if __name__ == "__main__":
    main()
