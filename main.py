#!/usr/bin/env python
from frontend.no_preload import NO_LD_PRELOAD_CTX

with NO_LD_PRELOAD_CTX():

    if __name__ == "__main__":
        import sys
        assert sys.version_info >= (3, 8), "Python 3.8+ required, got: {}".format(sys.version)

        from paritybench.main import main
        main()
