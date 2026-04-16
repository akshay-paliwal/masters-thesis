import torch
from Bio import SeqIO
from tqdm import tqdm
import os
import argparse

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig

# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Extract ESM-3 per-residue embeddings")
parser.add_argument("--input", type=str, required=True, help="Input FASTA file")
parser.add_argument("--output", type=str, required=True, help="Output directory")
parser.add_argument("--model", type=str, default="esm3-open", help="Model name")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESM3.from_pretrained(MODEL_NAME)
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
        protein = ESMProtein(sequence=sequence)
        protein_tensor = model.encode(protein)

        # Move tensor object to device if supported by current package version
        try:
            protein_tensor = protein_tensor.to(device)
        except AttributeError:
            pass

        # Forward pass
        # Forward pass
        with torch.no_grad():
            output = model.forward_and_sample(
                protein_tensor,
                SamplingConfig(return_per_residue_embeddings=True)
            )

        embedding = output.per_residue_embedding

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)

        embedding = embedding.cpu()

        # Remove BOS/EOS tokens if present
        if embedding.shape[0] == len(sequence) + 2:
            print(f"[INFO] Trimming BOS/EOS for {protein_id}")
            embedding = embedding[1:-1]

        # Safety check
        if embedding.shape[0] != len(sequence):
            print(
                f"[WARNING] Length mismatch for {protein_id}: "
                f"{embedding.shape[0]} vs {len(sequence)}"
            )

        # Save
        torch.save(
            {
                "embedding": embedding,
                "sequence": sequence,
                "protein_id": protein_id,
                "model": MODEL_NAME,
            },
            output_path,
        )

    except RuntimeError as e:
        print(f"[ERROR] Failed on {protein_id}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

    except Exception as e:
        print(f"[ERROR] Failed on {protein_id}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

print("✅ Extraction complete.")