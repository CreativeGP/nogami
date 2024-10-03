import argparse

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--memory", type=int, default=1000)
    args = argparse.parse_args()

    print(args.memory)
    a = [0 for i in range(args.memory)]
    while True:
        pass
