#!/usr/bin/env python
import sys

def main(n):
    n = int(n)
    for i, line in enumerate(sys.stdin):
        if i % n == 0:
            sys.stdout.write(line)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s n\n' % sys.argv[0])
        sys.exit(1)
    main(*sys.argv[1:])
