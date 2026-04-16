import torch
import esm
from Bio import SeqIO
from tqdm import tqdm
import os
import argparse

# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings")
parser.add_argument("--input", type=str, required=True, help="Input FASTA file")
parser.add_argument("--output", type=str, required=True, help="Output directory")
parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D")
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
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

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
        # Prepare input
        data = [(protein_id, sequence)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        # Forward pass
        with torch.no_grad():
            results = model(tokens, repr_layers=[33], return_contacts=False)

        token_embeddings = results["representations"][33]

        # Remove BOS/EOS tokens
        embedding = token_embeddings[0, 1:-1].cpu()

        # Safety check
        if embedding.shape[0] != len(sequence):
            print(f"[WARNING] Length mismatch for {protein_id}: "
                  f"{embedding.shape[0]} vs {len(sequence)}")

        # Save
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