import os
from argparse import ArgumentParser as AP

from os.path import expanduser, join

home = expanduser("~")
mlex = join(home, ".mlex")
fjobs = join(mlex, "jobs.txt")


def get_args():
    ap = AP()
    ap.add_argument("--job", "-j")
    return ap.parse_args()


def safe_mkdir(name):
    if not os.path.exists(name):
        os.mkdir(name)


def read(fname):
    with open(fname, "r") as f:
        return f.readlines()


def write(fname, data):
    with open(fname, "w") as f:
        f.writelines(data)


def add():
    args = get_args()

    data = read(fjobs)
    data.append(args.job)
    write(fjobs, data)


def main():

    if not os.path.exists(mlex):
        os.mkdir(mlex)

        # with open(fjobs, 'w') as f:
