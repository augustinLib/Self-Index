def main(argv=None):
    import argparse
    import os

    from selfindex.runtime.execution import launch
    from selfindex.runtime.platform.environment import ensure_env

    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--keys-out", required=True, metavar="PATH")
    args = parser.parse_args(argv)
    env = ensure_env()
    previous = os.getcwd()
    evolution = ["bcplus", args.tag, "--retriever", args.retriever]
    try:
        os.chdir(env.repo_root)
        return launch.main([*evolution, "--keys-out", args.keys_out])
    finally:
        os.chdir(previous)
