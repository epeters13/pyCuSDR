# Copyright: (c) 2021, Edwin G. W. Peters

import zmq
import numpy as np
import time




def send_to_sdr(NUM_TESTS = 1000, TIME_BETWEEN_PACKETS_MS = 1000, PACKET_LEN = 21,ADDR='tcp://127.0.0.1:5501'):

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(ADDR)

    t_tot = 0
    packets = []
    for i in range(NUM_TESTS):
        packets.append(np.mod(np.arange(PACKET_LEN),256).astype(np.uint8))

    for i in range(NUM_TESTS):

        print(f'sending packet {i}')
        p = np.concatenate(( np.array([0x40,0x00,0x00,0x01,0x17],dtype=np.uint8),packets[i],np.array([0x11,0x11],dtype=np.uint8)))
        # p[20] = 0
        sock.send(p)

        time.sleep(TIME_BETWEEN_PACKETS_MS/1000)


        
    sock.close()


if __name__ == "__main__":

    send_to_sdr()
