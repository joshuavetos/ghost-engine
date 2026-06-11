from ghost_engine.benchmark import run_benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--out", default="results")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--sampling", choices=["full", "first", "random", "stratified"])
    args = parser.parse_args()
    run_benchmark(args.config, args.out, max_cases=args.max_cases, sampling=args.sampling)
