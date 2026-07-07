import glob
import os
import subprocess


def valid_fchk(fchkfile):
    """Return True if the formatted checkpoint looks usable.

    ``formchk`` can exit 0 yet still write a 0-atom stub for some converged
    checkpoints; a return code check alone therefore does not prove success.
    We validate the output by parsing the ``Number of atoms`` header and
    requiring it to be greater than zero.
    """
    if not os.path.isfile(fchkfile) or os.path.getsize(fchkfile) == 0:
        return False
    try:
        with open(fchkfile, 'r') as fchk:
            for line in fchk:
                if line.startswith('Number of atoms'):
                    return int(line.split()[-1]) > 0
    except (OSError, ValueError):
        return False
    return False


def Get_chklist(remove):
    """Convert every ``./*.chk`` to ``.fchk`` with ``formchk``.

    The source ``.chk`` is only removed (when ``remove == 1``) once the produced
    ``.fchk`` has been validated, so a failed or empty conversion never destroys
    the original checkpoint. Returns the list of ``.chk`` files whose conversion
    failed (empty when every conversion succeeded).
    """
    failed = []
    for f in glob.glob('./*.chk'):
        print(f)
        fchkfile = f[:-4] + '.fchk' if f.endswith('.chk') else f + '.fchk'
        try:
            returncode = subprocess.run(['formchk', f]).returncode
        except Exception as e:
            print(f"Failed converting chk to fchk: {e}")
            returncode = -1
        if returncode != 0 or not valid_fchk(fchkfile):
            print(f"Failed converting chk to fchk! Keeping source checkpoint: {f}")
            failed.append(f)
            continue
        if remove == 1:
            os.remove(os.path.join('.', f))
        else:
            pass
    return failed


if __name__ == '__main__':
    Get_chklist()
