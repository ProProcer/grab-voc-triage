import sys
import json
import time
from pathlib import Path
from argparse import ArgumentParser

# Ensure root project directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
from openai import OpenAI
import config
import dotenv
dotenv.load_dotenv()

def parse_args():
    parser = ArgumentParser(description="End-to-end automated batch classification pipeline with chunking and resume support")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Path to system prompt markdown file")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to final output prediction CSV")
    parser.add_argument("-m", "--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--chunk_size", type=int, default=500, help="Number of rows per chunk (default: 500)")
    parser.add_argument("--poll_interval", type=int, default=30, help="Seconds to wait between status polls (default: 30)")
    return parser.parse_args()

def main():
    args = parse_args()
    client = OpenAI()

    prompt_path = Path(args.prompt)
    input_path = Path(args.input)
    model = args.model
    chunk_size = args.chunk_size

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (Path(config.MODEL_PREDS_DIR) / "openai" / (input_path.stem + f"__{prompt_path.stem}__{model}")).with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    df = pd.read_csv(input_path)
    total_rows = len(df)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    checkpoint_path = Path("data/interim") / f"checkpoint_{input_path.stem}__{prompt_path.stem}__{model}.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"Resuming from existing checkpoint: {len(checkpoint)}/{total_rows} rows already classified.")
        except Exception:
            checkpoint = {}

    print("=" * 60)
    print("STARTING BATCH PIPELINE")
    print(f"Input: {input_path} ({total_rows} rows)")
    print(f"Prompt: {prompt_path}")
    print(f"Model: {model}")
    print(f"Chunk size: {chunk_size} rows ({total_chunks} total chunks)")
    print(f"Output: {output_path}")
    print("=" * 60)

    for chunk_idx in range(total_chunks):
        start_row = chunk_idx * chunk_size
        end_row = min(start_row + chunk_size, total_rows)
        chunk_df = df.iloc[start_row:end_row]

        # Check if this chunk is already fully classified in checkpoint
        chunk_indices = [str(idx) for idx in chunk_df.index]
        if all(idx in checkpoint for idx in chunk_indices):
            print(f"Chunk {chunk_idx + 1}/{total_chunks} (rows {start_row}-{end_row - 1}) already completed in checkpoint. Skipping.")
            continue

        print(f"\n--- Submitting Chunk {chunk_idx + 1}/{total_chunks} (rows {start_row} to {end_row - 1}) ---")
        temp_jsonl = Path("data/interim") / f"batch_pipeline_chunk_{chunk_idx}.jsonl"

        with open(temp_jsonl, "w", encoding="utf-8") as f:
            for idx, row in chunk_df.iterrows():
                content = "" if pd.isna(row["content"]) else str(row["content"])
                task = {
                    "custom_id": f"row_{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "temperature": 0.0,
                        "max_completion_tokens": 150,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f'Review: "{content}"'}
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "defect_schema",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        c: {"type": "string", "enum": ["NEG", "ABSENT"]}
                                        for c in config.CATEGORIES
                                    },
                                    "required": list(config.CATEGORIES),
                                    "additionalProperties": False
                                }
                            }
                        }
                    }
                }
                f.write(json.dumps(task) + "\n")

        with open(temp_jsonl, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")

        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"Batch {batch_job.id} submitted for chunk {chunk_idx + 1}. Polling until completed...")

        # Poll until this chunk completes
        while True:
            time.sleep(args.poll_interval)
            batch = client.batches.retrieve(batch_job.id)
            counts_str = ""
            if batch.request_counts:
                counts_str = f"({batch.request_counts.completed}/{batch.request_counts.total} done)"
            print(f"Chunk {chunk_idx + 1} status: {batch.status} {counts_str}")

            if batch.status == "completed":
                break
            elif batch.status in ["failed", "expired", "cancelled"]:
                print(f"ERROR: Batch failed with status '{batch.status}'. Errors: {batch.errors}")
                return

        # Collect results
        output_content = client.files.content(batch.output_file_id).text
        for line in output_content.strip().split("\n"):
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = str(item.get("custom_id", "").replace("row_", ""))

            pred = {f"{c}_pred": "ERROR" for c in config.CATEGORIES}
            resp = item.get("response")
            if resp and resp.get("status_code") == 200:
                try:
                    content_str = resp["body"]["choices"][0]["message"]["content"]
                    parsed = json.loads(content_str)
                    if "DRIVER_OPS" not in parsed and "DRIVER_OPERATIONS" in parsed:
                        parsed["DRIVER_OPS"] = parsed["DRIVER_OPERATIONS"]
                    for c in config.CATEGORIES:
                        if c in parsed:
                            pred[f"{c}_pred"] = parsed[c]
                except Exception:
                    pass
            checkpoint[row_id] = pred

        # Save checkpoint & partial output CSV
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        formatted_checkpoint = {int(k) if k.isdigit() else k: v for k, v in checkpoint.items()}
        preds_df = pd.DataFrame.from_dict(formatted_checkpoint, orient="index").reindex(df.index)
        final_df = pd.concat([df, preds_df], axis=1)
        final_df.to_csv(output_path, index=False)

        print(f"Chunk {chunk_idx + 1}/{total_chunks} collected. Overall: {len(checkpoint)}/{total_rows} rows classified.")

    print("\n" + "=" * 60)
    print(f"ALL CHUNKS COMPLETE! Classified {len(checkpoint)}/{total_rows} rows.")
    print(f"Final predictions saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
