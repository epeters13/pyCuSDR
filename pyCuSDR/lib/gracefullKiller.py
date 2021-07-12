# Copyright: (c) 2021, Edwin G. W. Peters

import signal

class GracefulKiller:
    """
    Catch the sigQUIT signal and handle it gracefully to initiate a proper shutdown of the modem
    """
    def __init__(self):
        self.kill_now = False
        # signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        signal.signal(signal.SIGQUIT, self.exit_gracefully)

    def exit_gracefully(self,signum, frame):
        self.kill_now = True
