import os
import sys
import subprocess
import re


def ChargeSpin_fchk(fchkfile):
    TotalCharge = 0
    SpinMulti = 1
    with open(fchkfile, 'r') as fchk:
        lines = fchk.readlines()
    for line in lines:
        if (re.match("Charge", line)):
            Charge_inf = line.split()
            TotalCharge = int(Charge_inf[-1])
        if (re.match("Multiplicity", line)):
            Multi_inf = line.split()
            SpinMulti = int(Multi_inf[-1])
    lines = ""
    return TotalCharge, SpinMulti


def Get_fchk(jobname):
    """Convert ``{jobname}.fchk`` back to a binary checkpoint with ``unfchk``.

    Returns ``(TotalCharge, SpinMulti, ok)`` where ``ok`` is ``False`` when the
    conversion failed (``unfchk`` exited non-zero / crashed, or produced no
    usable ``.chk``). Callers can then skip work that depends on the
    reconstructed checkpoint instead of silently proceeding with a missing or
    invalid ``.chk``.
    """
    fchkfile = f"{jobname}.fchk"
    TotalCharge, SpinMulti = ChargeSpin_fchk(fchkfile)
    ok = True
    try:
        subprocess.run(['unfchk', fchkfile], check=True)
    except Exception as e:
        print(f"Failed converting fchk to chk: {e}")
        ok = False
    # unfchk can crash (e.g. segfault on a malformed fchk) without writing a
    # usable checkpoint, so verify the output actually exists and is non-empty.
    chkfile = f"{jobname}.chk"
    if not os.path.isfile(chkfile) or os.path.getsize(chkfile) == 0:
        print(f"fchk to chk conversion did not produce a usable checkpoint: {chkfile}")
        ok = False
    return TotalCharge, SpinMulti, ok


if __name__ == '__main__':
    usage = 'Usage; %s jobname' % sys.argv[0]
    try:
        jobname = sys.argv[1]
    except:
        print (usage); sys.exit()
    Charge, SpinMulti, ok = Get_fchk(jobname)
    print(f"Charge: {Charge}")
    print(f"SpinMulti: {SpinMulti}")
    print(f"Converted: {ok}")
