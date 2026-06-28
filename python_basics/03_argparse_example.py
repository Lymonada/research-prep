import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--type1", type=str, required=True)
    parser.add_argument("--type2", type=int, default=None)
    parser.add_argument("--height", type=str, required=True)
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--legendary", type=bool, default=False)

    return parser.parse_args()

def main():
    args = parse_args()

    print("Name:", args.name)
    print("Type1:", args.type1)
    print("Type2:", args.type2)
    print("Height:", args.height)
    print("Weight:", args.weight)
    print("Is Legendary?:", args.legendary)

# Only run when this file is executed -> only when I enter "python 03_argparse_example.py" in the terminal
if __name__ == "__main__":
    main()
