# Copyright: (c) 2021, Edwin G. W. Peters

import zmq
import numpy as np
import time

class Rx_uhf_from_sdr():

    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False
        
    def rx_uhf_from_sdr(self,ADDR='tcp://127.0.0.1:5502'):

        ctx = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.connect(ADDR)
        

        cnt = 0
        while self._running == True:
            t = time.time()
            try:
                d = sock.recv()
            except zmq.error.Again:
                continue
            ts = time.time()-t
            cnt += 1
            print(f'UHF: packet no {cnt} -- received {len(d)} bytes: {d}')



if __name__ == "__main__":

    cls = Rx_uhf_from_sdr()
    cls.rx_uhf_from_sdr()
