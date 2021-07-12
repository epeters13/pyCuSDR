# Copyright: (c) 2021, Edwin G. W. Peters

import numpy as np
import string
from lib.safe_lists import *

LOG_NAME = "pyCuSDR"

"""
In benchmark mode, only packets from one channel are sent to the link manager
Used in:
   decoder_process
"""

BENCHMARK_MODE = False
SAVETX_DATA = False  # TX save the last modulated packet signal in npy file
STORE_BITS_IN_FILE = False # Rx store received data in file (can get slow)

"""
TRUST:
   normal: 2 * trustweight
   symbol error: -1
   clipping:  -2
"""
TRUSTTYPE = np.int8
DATATYPE = np.int8

MODULATORDTYPE = np.complex64

printableChars = set(string.printable)

printBytesAsHex = lambda x: ' '.join(['{:02X}'.format(i) for i in x])



def attr_in_config(cfg,attr,defaultVal,warningOnFail = True):
    """
    Method to check if attribute exists in config and use default if not available
    
    Inputs:
       cfg -- config file including section (ex. foo['Main'])
       attr -- attribute string (ex. 'bar')
       defaultVal -- default value if 'attr' not found in cfg
       warningOnFail -- print warning if 'attr' not found in cfg (default: True)
    Returns:
       cfg[attr] or defaultVal
    """
    
    if attr in cfg.keys():
        return cfg[attr]
    else:
        if warningOnFail:
            log.warning(f'\'{attr}\' not specified in config. Using default value of {defaultVal}')
        return defaultVal


    
    
def json_str_list_to_int_list(json_list,json_number_base=16):
    """
    returns a json list to a list of ints with base 'json_number_base' [default 16]

    """
    return [int(k,json_number_base) for k in json_list]
    
