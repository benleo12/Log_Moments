import argparse

def main():
    parser = argparse.ArgumentParser(description="Test Parser")
    parser.add_argument('--n_step', type=int, default=100, help='Number of steps in the process')
    args = parser.parse_args()
    print("n_step:", args.n_step)

if __name__ == "__main__":
    main()

