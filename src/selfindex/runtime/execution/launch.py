import argparse


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("domain")
    parser.add_argument("tag")
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--keys-out", metavar="PATH")
    args = parser.parse_args(argv)
    from selfindex.pipeline.run import main as run

    keys_out = ["--keys-out", args.keys_out] if args.keys_out else []
    return run([args.domain, args.tag, "--retriever", args.retriever, *keys_out])
