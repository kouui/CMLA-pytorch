import sys
from collections import OrderedDict

if __name__ == "__main__":

    inFile = sys.argv[1]
    outFile = sys.argv[2]

    with open(inFile, 'r') as f:
        lns = f.readlines()

    data = OrderedDict()
    for ln in lns:

        if ln[0].isnumeric():
            continue

        words = ln.split(':')
        try:
            data[words[0]].append( float(words[1]) )
        except KeyError:
            data[words[0]] = []
            data[words[0]].append( float(words[1]) )
    with open(outFile, 'w') as f:

        Ls = []
        f.write("epoch  ")
        for key in data.keys():
            f.write(f"{key:20s}")
            Ls.append(len(data[key]))
        f.write('\n')

        for L in Ls[1:]:
            assert L==Ls[0]

        for i in range(L):
            f.write(f"{i+1:04d}   ")
            for key, value in data.items():
                s_ = f"{value[i]:<.4f}"
                f.write(f"{s_:<20s}")
            f.write('\n')
