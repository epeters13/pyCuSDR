# Original Author    : Edwin G. W. Peters @ sdr-Surface-Book-2
#   Creation date    : Sat Jul 10 08:26:04 2021 (+1000)
#   Email            : edwin.peters@unsw.edu.au
# ------------------------------------------------------------------------------
# Last-Updated       : Mon Jul 12 15:37:11 2021 (+1000)
#           By       : Edwin G. W. Peters @ sdr-Surface-Book-2
# ------------------------------------------------------------------------------
# File Name          : rig_server.py
# Description        : 
# ------------------------------------------------------------------------------
# Copyright          : Insert license
# ------------------------------------------------------------------------------


import threading
import socket
import string
from enum import Enum
from lib.freq_from_rangerate import *


class Response_codes(Enum):
    SUCCESS = 0
    UNIMLEMENTED = -4
    PARAM = -1
    PROTOCOL = -8


SHORT_TO_LONG_CMD ={
    b'f' :     b'\get_freq' ,
    b'F' :     b'\set_freq' ,
    b'I' :     b'\set_split_freq',
    b'i' :     b'\get_split_freq',
    b'S' :     b'\set_split_vfo',
    b's' :     b'\get_split_vfo',
    b'V' :     b'\set_vfo' ,
    b'v' :     b'\get_vfo' ,
    b'M' :     b'\set_mode' ,
    b'm' :     b'\get_mode' ,
    b'X' :     b'\set_split_mode',
    b'x' :     b'\get_split_mode',
}

dump = """0\n2\n2\n150000.000000 30000000.000000  0x900af -1 -1 0x10 000003 0x3\n0 0 0 0 0 0 0\n150000.000000 30000000.000000  0x900af -1 -1 0x10 000003 0x3\n0 0 0 0 0 0 0\n0 0\n0 0\n0\n0\n0\n0\n\n\n0x0\n0x0\n0x0\n0x0\n0x0
"""

class Rig_server(threading.Thread):


    def __init__(self,sock,addr, radios_rx, radios_tx):
        """
        Provides a control interface for hamlib
        """
        threading.Thread.__init__(self)
        self.daemon = True

        self.addr = addr
        self.sock = sock
        self.sock.settimeout(0.5)

        if len(radios_rx) > 0:
            self.radios_rx = radios_rx
        else:
            self.radios_rx = [DummyRadio()]

        if len(radios_tx) > 0:            
            self.radios_tx = radios_tx
        else:
            self.radios_tx = [DummyRadio()]
            
        self.running = True
        self.rx_buf = b''

        # state of this radio
        self._mode = 'CW'
        self._vfo = 'VFOA'


        print(f'Client from {self.addr}')

    def _send(self,data_bs):
        # print(f'data_bs {data_bs}')
        try:
            self.sock.sendall(data_bs)
        except socket.error:
            self.sock.close()
            self.running = False # quit

    def send_response(self,data):
        if isinstance(data,Enum): # response code
            data_bs = f'RPRT {data.value}\n'.encode('ascii')
        else: # data response
            data_bs = f'{data}\n'.encode('ascii')
        self._send(data_bs)


    def parse_commands(self,cmd,val):

        # if cmd in uppercase: # setter
        if b'\set' in cmd: # setter
            if cmd == b'\set_freq':
                self.rx_freq = float(val)
                self.send_response(Response_codes.SUCCESS)
            elif cmd == b'\set_split_freq':
                # todo: shall we make this robust for double arguments in case VFO is provided?  The \\chk_vfo response should have taken care of this
                self.tx_freq = float(val)
                self.send_response(Response_codes.SUCCESS)
            elif cmd == b'\set_rangerate':
                self.rangerate = float(val)
            else:
                self.send_response(Response_codes.UNIMLEMENTED)
                                

        else:
            if cmd == b'\get_freq':
                self.send_response(self.rx_freq)
            elif cmd == b'\get_split_freq':
                self.send_response(self.tx_freq)
            elif cmd == b'\get_vfo':
                self.send_response(self.vfo)
            elif cmd == b'\get_split_vfo':
                self.send_response(f'0 {self.vfo}')
            elif cmd == b'\get_rangerate':
                self.send_response(self.rangerate)
            else:
                self.send_response(Response_codes.UNIMLEMENTED)



    def check_for_commands(self):
        
        if b'\n' in self.rx_buf: # commands are terminated with '\n'
            # print(f'\n\nrx_buf {self.rx_buf}')
            cmd_full, self.rx_buf = self.rx_buf.split(b'\n',1)
            # print(f'cmd_full {cmd_full}')
            if cmd_full.startswith(b'\\'): # special commands, these are treated separately
                if b'\\chk_vfo' in cmd_full:
                    self.send_response('CHKVFO 0')
                elif b'\\dump' in cmd_full:
                    self.send_response(dump)
                else:
                    self.send_response(Response_codes.UNIMLEMENTED)
            elif cmd_full.startswith(b'\g') or cmd_full.startswith(b'\s'): # long commands
                cmd,data = cmd_full.split(b' ',1)


                if cmd == b'':
                    self.send_response(Response_codes.UNIMLEMENTED)
                else:
                    self.parse_commands(cmd,val)

            else:
                if len(cmd_full) > 1:
                    cmd,val = cmd_full.split(b' ',1)
                else:
                    cmd = cmd_full
                    val = b'0'
                if cmd == b'':
                    self.send_response(Response_codes.UNIMLEMENTED)
                else:
                    cmd = SHORT_TO_LONG_CMD.get(cmd,b'')
                    # print(f'cmd {cmd}')
                    self.parse_commands(cmd,val)
                
                

    def run(self):

        while self._running_state == True:
            try:
                t = self.sock.recv(1024)
            except socket.timeout:	# This does not work
                pass
            except socket.error:	# Nothing to read
                pass
            else:					# We got some characters
                self.rx_buf += t

            if not t:
                break
            self.check_for_commands()
            
        print(f'socket from {self.addr[0]}:{self.addr[1]} closed')

    def terminate(self):
        self._running_state = False
        # self.socket.close()
        
    @property
    def rx_freq(self,idx=0):
        return self.radios_rx[idx].freq_hl 

    @rx_freq.setter
    def rx_freq(self,val,idx=0):
        # Do we want this?
        # ideally this is a reference frequency and the sdr can report the error
        self.radios_rx[idx].freq_hl = val
        print(f'rx rangerate {self.radios_rx[0].rangerate}')

    @property
    def tx_freq(self,idx=0):
        return self.radios_tx[idx].freq_hl 

    @tx_freq.setter
    def tx_freq(self,val,idx=-1):
        # the radio takes care of extracting the Doppler based on the center frequency in the config
        
        if idx >= 0:
            # set desired channel
            self.radios_tx[idx].freq_hl = val
        else:
            # set all channels -- they get range rate out of this
            for r in self.radios_tx:
                r.freq_hl = val
                
    @property
    def running(self):
        return self._running_state

    @running.setter
    def running(self,newstate):
        self._running_state = newstate


    @property
    def vfo(self):
        return self._vfo

    @vfo.setter
    def vfo(self,val):
        self._vfo = val.strip()



class DummyRadio():

    """
    Just a dummy class implementing the methods adressed by rig_server
    """


    def __init__(self):

        self._Fc = 186e6 # set from config
        self._rangerate = 0
        self._doppler = 0 


    @property
    def freq_hl(self):
        # return frequency for hamlib
        return self.Fc + self.doppler

    @freq_hl.setter
    def freq_hl(self,val):
        # extract the rangerate from the frequency provided by hamlib
        self.rangerate = rangerate_from_freq(val,self.Fc)

    @property
    def Fc(self):
        return self._Fc

    @Fc.setter
    def Fc(self,val):
        self._Fc = val

    @property
    def rangerate(self):
        return self._rangerate

    @rangerate.setter
    def rangerate(self,val):
        self.doppler = val*self.Fc/scipy.constants.speed_of_light
        self._rangerate = val

    @property
    def doppler(self):
        return self._doppler

    @doppler.setter
    def doppler(self,val):
        self._doppler = val

