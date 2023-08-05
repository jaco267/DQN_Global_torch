import sys
import gzip
import bz2

spec_verbose = 0
via_cost = 1


###############################################################################
# Process all the switches
# -h help
# -H noheader
# -n net name for an eps file
# -v verbose level

def process_switches(args):
    global spec_verbose
    global via_cost

    opt_h = False
    opt_H = False
    opt_n = None
    opt_v = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '-h':
            opt_h = True
        elif arg == '-H':
            opt_H = True
        elif arg == '-n':
            i += 1
            opt_n = args[i]
        elif arg == '-v':
            i += 1
            opt_v = int(args[i])
            spec_verbose = 1 if opt_v > 1 else 0
        i += 1

    if len(args) != 3 or opt_h:
        print("Usage: {} [input design] [routed result]".format(args[0]))
        print("Options: -n [net name]    Make net.ps file")
        print("         -v [level]       Verbosity level (0-2)")
        print("         -h               this (help) message")
        print("         -H               do not print header")
        print("Notes: Tot OF - Total overflow")
        print("       Max OF - Maximum overflow")
        print("       WL     - Total wirelength of the solution")
        sys.exit(0)

    return opt_n, opt_H, opt_v

opt_n, opt_H, opt_v = process_switches(sys.argv)

inname = sys.argv[1]
routename = sys.argv[2]

innameO = inname
routenameO = routename

if inname.endswith('.gz'):
    inname = gzip.open(inname, 'rt')
elif inname.endswith('.bz2'):
    inname = bz2.open(inname, 'rt')
else:
    inname = open(inname, 'r')

if routename.endswith('.gz'):
    routename = gzip.open(routename, 'rt')
elif routename.endswith('.bz2'):
    routename = bz2.open(routename, 'rt')
else:
    routename = open(routename, 'r')