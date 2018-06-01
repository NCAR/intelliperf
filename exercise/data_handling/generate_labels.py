import os
import sys
import math
import csv
import argparse
import shutil

num_rows = 999

def main():

    # parse arguments
    parser = argparse.ArgumentParser(description='Generating labels in folding csv data file')
    parser.add_argument('csvfile', metavar='csvfile', type=str, help='Folding csv file path')
    parser.add_argument('-r', '--range', dest='ranges', type=str, action='append',
        default=[], help='%% range of elapsed time to be labelled as "1"')
    parser.add_argument('-o', '--output', dest='output', type=str,
        default=None, help='path to output file')

    args = parser.parse_args()

    if not os.path.exists(args.csvfile):
        raise Exception("Input file does not exist: %s"%args.csvfile)

    # collect ranges
    # generate labels

    labels = ["-1.0" for _ in range(num_rows)]

    for commaranges in args.ranges:
        ranges = commaranges.split(',') 
        for dashrange in ranges:
            begin, end = map(lambda x: int(math.floor(float(x)*10.0)), dashrange.split('-'))
            if begin < 0 or end > 1000:
                raise Exception("Wrong range: %s"%dashrange)
            end = min(num_rows, end)

            for idx in range(begin, end):
                labels[idx] = "1.0"

    # read csv file
    module = None
    region = None

    with open(args.csvfile,'r') as f:
        plots=csv.reader(f,delimiter=';')
        module, region, _, _, _ = next(plots)

    if args.output is not None:
        outfile = args.output
    else:
        outfile = "%s.labelled%s"%os.path.splitext(args.csvfile)
    shutil.copy(args.csvfile, outfile)

    with open(outfile, "a") as f:
        for idx in range(num_rows):
            row = [] 
            row.append(module)
            row.append(region)
            row.append("LABEL")
            row.append("{0:.3f}".format(float(idx)*0.001+0.001))
            row.append(labels[idx])
            f.write(";".join(row)+"\n")
    #import pdb; pdb.set_trace()
    # write new csv file

    return 0

if __name__ == "__main__":
    sys.exit(main())
