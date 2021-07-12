# Original Author    : Edwin G. W. Peters @ sdr-Surface-Book-2
#   Creation date    : Sat Jul 10 07:55:14 2021 (+1000)
#   Email            : edwin.peters@unsw.edu.au
# ------------------------------------------------------------------------------
# Last-Updated       : Mon Jul 12 15:53:20 2021 (+1000)
#           By       : Edwin G. W. Peters @ sdr-Surface-Book-2
# ------------------------------------------------------------------------------
# File Name          : start_py_cu_sdr_hamlib.py
# Description        : 
# ------------------------------------------------------------------------------
# Copyright          : Insert license
# ------------------------------------------------------------------------------


import socket
import sys
import argparse
import pyCuSDR
import time
from pyCuSDR import VERSION

from pyLoadModularJson import loadModularJson
import rig_server

# for testing
sys.path.append('../test/')
from dummy_radios import Radio

CFG_FILE = "../config/hamlib_sockets.json"

"""
Listen on hamlib sockets from config file.

When socket opened wait for first frequency setting?
    This can later be used for GRC and SDR center frequency retuning
 
Start SDR with the config and parameters
"""


parser = argparse.ArgumentParser(prog='pyCu-SDR', description = 'Software defined radio that performs Doppler search and supports multiple receiving stations and antennas')

parser.add_argument('-c', '--configFile', help='Configuration file to use', action='store', default = CFG_FILE)
parser.add_argument('-v',help=' -vv, -vvv increases verbosity',action='count',default=0)
parser.add_argument('-V','--version',help='Show program version', action = 'version',version = '%(prog)s ' + str(VERSION))


args = parser.parse_args()

cfg_hamlib = loadModularJson(args.configFile)

radio_names = cfg_hamlib.keys()
sockets = {}
for r in radio_names:
    print(f'Opening socket for {r} on {cfg_hamlib[r]["addr"]}:{cfg_hamlib[r]["port"]}')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((cfg_hamlib[r]['addr'],cfg_hamlib[r]['port']))
    sock.settimeout(0.1)
    sock.listen(0)
    sockets[r] = sock

    

# loop waiting for clients
while True:

    for radio_name in sockets.keys():
        sock = sockets[radio_name]

        try:
            connection, client_address = sock.accept()
        except socket.error:
            pass
        else:
            print(f'Got connection from {client_address[0]}:{client_address[1]}')
            ## Start the radio
            # start new config file for each pass.
            # pycu_sdr(config_name)
            ## Dummy radios
            sdr = pyCuSDR.PyCuSDR(cfg_hamlib[radio_name]['config'],args)
            
            radios_rx = sdr.demodulators
            radios_tx = sdr.modulators
            rs = rig_server.Rig_server(connection,client_address,radios_rx,radios_tx)
            rs.start()
            sdr.start()

            try:
                while sdr.is_alive() & rs.is_alive():
                    time.sleep(0.1)

                # print(f'sdr alive {sdr.is_alive()}')
                # print(f'rig server alive {rs.is_alive()}')
            except Exception as e:
                print(e)
                
            finally:
                rs.terminate()
                print(f'terminating SDR')
                sdr.terminate()
                print(f'closing connection')
                connection.close()
                print(f'connection closed')
                rs.join()
                print(f'joined')

                print(f'Finished {client_address[0]}:{client_address[1]}')

            
    
for k in sockets.keys():
    sockets[k].close()

    
