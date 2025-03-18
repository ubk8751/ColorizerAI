import argparse
from src.train import train
from src.inference import run_inference
from src.dataset import get_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train, run inference, or prepare dataset for ColorizationNet")
    parser.add_argument("mode", choices=["train", "inference", "prepare-data"], help="Choose an operation mode")
    parser.add_argument("--input-image", type=str, help="Provide input image name for inference", default=None)

    args = parser.parse_args()

    if args.mode == "prepare-data":
        print("Downloading and preparing dataset...")
        get_dataloader()  # This will download and prepare dataset
        print("Dataset ready.")
    
    elif args.mode == "train":
        print("Running training...")
        train()

    elif args.mode == "inference":
        if args.input_image is None:
            print("Error: Please provide an image to colorize using --input-image")
            exit(1)
        print(f"Running inference on {args.input_image}...")
        run_inference(img_name=args.input_image)

if __name__ == "__main__":
    main()
