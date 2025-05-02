#!/usr/bin/env python3
"""
mpigdb.py – one-prompt GDB wrapper for MPI ranks
"""
import argparse, os, shlex, sys, pexpect


class GDBSession:
    PROMPT = r"\(gdb\)\s?$"

    def __init__(self, rank, cmd, *, mi=False, env=None):
        if mi:
            self.PROMPT = r"\(gdb\)"
        self.rank = rank
        self.proc = pexpect.spawn(
            cmd[0], cmd[1:], env=env, encoding="utf-8", echo=False, timeout=None
        )
        pats = [self.PROMPT, r"Enable debuginfod .* \[n\]\)\s?$"]
        while True:
            i = self.proc.expect(pats, timeout=30)
            if i == 0:
                break
            if i == 1:
                self.proc.sendline("n")

    def send(self, line):
        self.proc.sendline(line)

    def query(self, line):
        self.proc.sendline(line)
        self.proc.expect(self.PROMPT)
        return self.proc.before.strip()

    def close(self):
        try:
            self.proc.sendline("quit")
            self.proc.expect(pexpect.EOF, 3)
        except Exception:
            pass
        self.proc.close(force=True)


class MPIController:
    def __init__(self, n, exe, args, *, gdb="gdb", mi=False):
        base = [
            gdb,
            "--quiet",
            "-iex",
            "set debuginfod enabled off",
            "-ex",
            "set pagination off",
            "--args",
            exe,
            *args,
        ]
        if mi:
            base.insert(2, "--interpreter=mi2")
        self.ranks = []
        for r in range(n):
            env = os.environ.copy()
            env["MPI_RANK"] = str(r)
            self.ranks.append(GDBSession(r, base, mi=mi, env=env))

    def _fan(self, fn, line, show=False):
        for s in self.ranks:
            out = fn(s, line)
            if show:
                print(f"[rank {s.rank}]")
                print(out)

    def send_all(self, line):
        self._fan(lambda s, l: s.send(l), line)

    def query_all(self, line):
        self._fan(lambda s, l: s.query(l), line, True)

    def send_rank(self, r, line):
        self.ranks[r].send(line)

    def query_rank(self, r, line):
        print(self.ranks[r].query(line))

    def info(self):
        [print(f"rank {s.rank} → pid {s.proc.pid}") for s in self.ranks]

    def shutdown(self):
        [s.close() for s in self.ranks]


EXEC_CMDS = {
    "run",
    "r",
    "continue",
    "c",
    "start",
    "next",
    "n",
    "step",
    "s",
    "finish",
    "jump",
    "signal",
}


def repl(ctrl):
    try:
        while True:
            l = input("(mpigdb) ").strip()
            if not l:
                continue
            if l in {"quit", "exit"}:
                break
            if l == "list":
                ctrl.info()
                continue
            if l.startswith("all? "):
                ctrl.query_all(l[5:])
                continue
            if l.startswith("all "):
                cmd = l[4:].split()[0]
                if cmd in EXEC_CMDS:
                    ctrl.send_all(l[4:])
                else:
                    ctrl.query_all(l[4:])
                continue
            if l.startswith("r "):
                toks = shlex.split(l)
                if len(toks) < 3:
                    print("usage: r <rank> <gdb-cmd>")
                    continue
                rank, rest = int(toks[1]), " ".join(toks[2:])
                if rest.split()[0] in EXEC_CMDS:
                    ctrl.send_rank(rank, rest)
                else:
                    ctrl.query_rank(rank, rest)
                continue
            print("prefix with  all / all? / r <rank>  or type  list")
    finally:
        ctrl.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--np", type=int, required=True)
    ap.add_argument("--gdb", default="gdb")
    ap.add_argument("--mi", action="store_true")
    ap.add_argument("exe")
    ap.add_argument("exe_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    repl(MPIController(a.np, a.exe, a.exe_args, gdb=a.gdb, mi=a.mi))


if __name__ == "__main__":
    main()
