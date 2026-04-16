import torch
from transformers import T5Tokenizer, T5EncoderModel
from Bio import SeqIO
from tqdm import tqdm
import os
import argparse

# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Extract ProtT5 embeddings")
parser.add_argument("--input", type=str, required=True, help="Input FASTA file")
parser.add_argument("--output", type=str, required=True, help="Output directory")
parser.add_argument("--model", type=str, default="Rostlab/prot_t5_xl_uniref50")
parser.add_argument("--limit", type=int, default=None, help="Limit number of sequences (for testing)")
args = parser.parse_args()

INPUT_FASTA = args.input
OUTPUT_DIR = args.output
MODEL_NAME = args.model

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load model
# -------------------------
print(f"Loading model: {MODEL_NAME}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
model = T5EncoderModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"Using device: {device}")

# -------------------------
# Load sequences
# -------------------------
records = list(SeqIO.parse(INPUT_FASTA, "fasta"))

if args.limit is not None:
    records = records[:args.limit]
    print(f"[INFO] Running on subset: {len(records)} sequences")

print(f"Total sequences: {len(records)}")

# -------------------------
# Helper function
# -------------------------
def preprocess_sequence(seq):
    # ProtT5 expects space-separated amino acids
    return " ".join(list(seq))

# -------------------------
# Processing loop
# -------------------------
for record in tqdm(records):
    protein_id = record.id
    sequence = str(record.seq)

    output_path = os.path.join(OUTPUT_DIR, f"{protein_id}.pt")

    # Skip if already processed
    if os.path.exists(output_path):
        continue

    try:
        # Preprocess sequence
        processed_seq = preprocess_sequence(sequence)

        # Tokenize
        inputs = tokenizer(processed_seq, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        embedding = outputs.last_hidden_state[0].cpu()  # (seq_len, 1024)

        # Safety check
        if embedding.shape[0] != len(sequence):
            print(f"[WARNING] Length mismatch for {protein_id}: "
                  f"{embedding.shape[0]} vs {len(sequence)}")

        # Save (same structure as ESM for consistency)
        torch.save({
            "embedding": embedding,
            "sequence": sequence,
            "protein_id": protein_id,
            "model": MODEL_NAME
        }, output_path)

    except RuntimeError as e:
        print(f"[ERROR] Failed on {protein_id}: {e}")
        torch.cuda.empty_cache()
        continue

print("✅ Extraction complete.")