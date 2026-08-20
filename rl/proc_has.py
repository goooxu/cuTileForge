#!/usr/bin/env python3
"""Print the pid of a live process whose cmdline contains argv[1].

Skips this scanner (the needle is on our own argv). Used by keep_grpo_alive
instead of an inline python -c blob whose quoting broke the remote check.
Do not pkill -f; this only looks.
"""
import os
import sys

if len(sys.argv) < 2:
    sys.exit(2)
needle = sys.argv[1]
me = str(os.getpid())
for pid in os.listdir("/proc"):
    if pid == me or not pid.isdigit():
        continue
    try:
        cmd = open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\0", b" ")
        cmd = cmd.decode("utf-8", "replace")
    except OSError:
        continue
    if needle in cmd:
        print(pid)
        sys.exit(0)
sys.exit(1)
