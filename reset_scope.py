"""Restore the scope to normal free-running operation.

Run this if a comparison/measurement left the scope in NORMal trigger, Average
mode, an alternate trigger source, or stopped — i.e. the display looks frozen or
doesn't update as usual.

    python reset_scope.py            # restore with AUTO trigger on CH1
    python reset_scope.py 2          # ... on CH2
"""

import sys

from scope import Scope

if __name__ == "__main__":
    ch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    s = Scope()
    s.restore(channel=ch)
    s.close()
