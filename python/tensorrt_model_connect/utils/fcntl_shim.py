"""Cross-platform compatibility shim for fcntl."""

import sys

try:
    import fcntl

    LOCK_EX = fcntl.LOCK_EX
    LOCK_NB = fcntl.LOCK_NB
    LOCK_UN = fcntl.LOCK_UN
    LOCK_SH = fcntl.LOCK_SH
    flock = fcntl.flock
except ImportError:
    if sys.platform == "win32":
        import msvcrt
        import os

        LOCK_EX = 2
        LOCK_SH = 1
        LOCK_NB = 4
        LOCK_UN = 8

        def flock(fd, operation):
            if hasattr(fd, "fileno"):
                fd = fd.fileno()
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                if operation & LOCK_UN:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                elif operation & LOCK_NB:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            except OSError as e:
                if operation & LOCK_NB:
                    raise BlockingIOError("Resource temporarily unavailable") from e
                raise
    else:
        raise
