# Copyright: (c) 2021, Edwin G. W. Peters

import sys
import threading
import time
sys.path.append('../zmq_listeners')

from zmq_send_tx_to_sdr import send_to_sdr
from zmq_recv_sdr_UHF_data import Rx_uhf_from_sdr
from zmq_recv_sdr_SBAND_data import Rx_sband_from_sdr


NUM_PACKETS = 10
TIME_BETWEEN_PACKETS_MS = 1000 # milisecond

tx_t = threading.Thread(target = send_to_sdr,kwargs={'NUM_TESTS' :NUM_PACKETS, 'TIME_BETWEEN_PACKETS_MS' : TIME_BETWEEN_PACKETS_MS})

rx_uhf_cls = Rx_uhf_from_sdr()
rx_uhf_t = threading.Thread(target = rx_uhf_cls.rx_uhf_from_sdr)

rx_sband_cls = Rx_sband_from_sdr()
rx_sband_t = threading.Thread(target = rx_sband_cls.rx_sband_from_sdr)

rx_cls_all = [rx_uhf_cls,rx_sband_cls]

rx_threads = [rx_uhf_t]
threads = [tx_t] + rx_threads

for t in threads:
    t.start()

# while tx_t.is_alive:
#     time.sleep(1)
tx_t.join()
print('tx finished')
time.sleep(1)

for c in rx_cls_all:
    c.terminate()

for t in rx_threads:
    t.join()
