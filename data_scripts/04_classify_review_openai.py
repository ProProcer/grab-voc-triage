import dotenv
dotenv.load_dotenv()
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from src.models.openai import OpenAIModel
import config

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-p", "--prompt", type = str)
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type = str, default = None)
    parser.add_argument("-m", "--model", type = str, default = "gpt-4o-mini")
    args = parser.parse_args()
    args.prompt = Path(args.prompt)
    args.input = Path(args.input)
    if args.output:
        args.output = Path(args.output)
    else:
        args.output = (Path(config.MODEL_PREDS_DIR) / "openai" / (args.input.stem + f"__{args.prompt.stem}__{args.model}")).with_suffix(".csv")
    return args

def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    with open(args.prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    model = OpenAIModel(system_prompt, args.model)

    preds = model.batch_classify(df["content"])

    preds_df = pd.DataFrame(preds)
    preds_df = preds_df.rename({
        c : c + "_pred" for c in preds_df.columns
    }, axis = 1)
    final_df = pd.concat((df, preds_df), axis =1)
    final_df.to_csv(args.output, index = False)

    

    
if __name__ == "__main__":
    main()
