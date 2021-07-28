from __future__ import absolute_import, unicode_literals, division, print_function

import argparse
import pandas as pd


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="data/History_CZV120_20210307_20210602.csv"
    )
    args_parsed = parser.parse_args(args_list)
    return args_parsed


def main():
    args = parse_args()
    data_path = args.data_path

    data_df = pd.read_csv(data_path, delimiter="|")

    return 0


if __name__ == "__main__":
    main()
