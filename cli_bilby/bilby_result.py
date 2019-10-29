""" A command line interface to ease the process of batch jobs on result files

Examples
--------
To convert all the JSON result files in `outdir` to hdf5 format:

    $ bilby_result -r outdir/*json -c hdf5

Note, by default this will save the new files in the outdir defined within the
results files. To give a new location, use the `--outdir` argument.

To print the version and `log_evidence` for all the result files in outdir:

    $ bilby_result -r outdir/*json --print version log_evidence

To generate a corner plot for all the files in outdir

    $ bilby_result -r outdir/*json --call plot_corner

This is effectively calling `plot_corner()` on each of the result files
individually. Note that passing extra commands in is not yet implemented.

"""
import argparse
import pandas as pd

import bilby
from bilby.core.utils import tcolors


def setup_command_line_args():
    parser = argparse.ArgumentParser(
        description="Helper tool for bilby result files")
    parser.add_argument(
        "results", nargs='?',
        help="List of results files.")
    parser.add_argument(
        "-r", "--results", nargs='+', dest="option_results", default=list(),
        help="List of results files (alternative to passing results as a positional argument).")
    parser.add_argument("-c", "--convert", type=str, choices=['json', 'hdf5'],
                        help="Convert all results.", default=False)
    parser.add_argument("-m", "--merge", action='store_true',
                        help="Merge the set of runs, output saved using the outdir and label")
    parser.add_argument("-o", "--outdir", type=str, default=None,
                        help="Output directory.")
    parser.add_argument("-l", "--label", type=str, default=None,
                        help="New label for output result object")
    parser.add_argument("-b", "--bayes", action='store_true',
                        help="Print all Bayes factors.")
    parser.add_argument("-p", "--print", nargs='+', default=None, dest="keys",
                        help="Result dictionary keys to print.")
    parser.add_argument("--call", nargs='+', default=None,
                        help="Result dictionary methods to call (no argument passing available).")
    parser.add_argument("--ipython", action='store_true',
                        help=("For each result given, drops the user into an "
                              "IPython shell with the result loaded in"))
    args = parser.parse_args()

    if args.results is None:
        args.results = []
    if isinstance(args.results, str):
        args.results = [args.results]
    args.results += args.option_results

    if len(args.results) == 0:
        raise ValueError("You have not passed any results to bilby_result")

    return args


def read_in_results(filename_list):
    results_list = []
    for filename in filename_list:
        results_list.append(bilby.core.result.read_in_result(filename=filename))
    return bilby.core.result.ResultList(results_list)


def print_bayes_factors(results_list):
    print("\nPrinting Bayes factors:")
    N = len(results_list)
    for i, res in enumerate(results_list):
        print("For label={}".format(res.label))
        index = ['noise'] + [results_list[j].label for j in range(i + 1, N)]
        data = [res.log_bayes_factor]
        data += [res.log_evidence - results_list[j].log_evidence for j in range(i + 1, N)]
        series = pd.Series(data=data, index=index, name=res.label)
        print(series)


def drop_to_ipython(results_list):
    for result in results_list:
        message = "Opened IPython terminal for result {}".format(result.label)
        message += "\nBilby result loaded as `result`"
        import IPython
        IPython.embed(header=message)


def print_matches(results_list, args):
    for r in results_list:
        print("\nResult file: {}/{}".format(r.outdir, r.label))
        for key in args.keys:
            for attr in r.__dict__:
                if key in attr:
                    print_line = [
                        "  ", tcolors.KEY, attr, ":", tcolors.VALUE,
                        str(getattr(r, attr)), tcolors.END]
                    print(" ".join(print_line))


def main():
    args = setup_command_line_args()
    results_list = read_in_results(args.results)
    if args.convert:
        for r in results_list:
            r.save_to_file(extension=args.convert, outdir=args.outdir)
    if args.keys is not None:
        print_matches(results_list, args)
    if args.call is not None:
        for r in results_list:
            for call in args.call:
                getattr(r, call)()
    if args.bayes:
        print_bayes_factors(results_list)
    if args.ipython:
        drop_to_ipython(results_list)
    if args.merge:
        result = results_list.combine()
        if args.label is not None:
            result.label = args.label
        if args.outdir is not None:
            result.outdir = args.outdir
        result.save_to_file()
