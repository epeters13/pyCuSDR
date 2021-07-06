# Copyright: (c) 2021, Edwin G. W. Peters

"""
The main file to start the sdr
Data is fed in over ZMQ from GNU radio, rtl SDR or bladeRF
"""

from __global__ import *


import sys, os
import time
import argparse
import signal

# modem modules
import demodulator_process
import decoder_process
import softCombiner
import modulator_process
import rpcInterface
import stats
import protocol
import lib
from pyLoadModularJson import loadModularJson
import atexit
import subprocess, re # for version tagging
import urllib, urllib.error, json # to get passID
import socket # to get hostname for GSID

# logging
import logging, logging.handlers
import multiprocessing_logger


LOG_FOLDER = '../logs'

LOG_FORMAT		= '%(asctime)s.%(msecs)03d %(module)-16s -- %(message)s '
LOG_FORMAT_DEBUG	= '%(asctime)s.%(msecs)03d %(levelname)s %(module)-16s %(funcName)-22s:%(lineno)5s -- %(message)s'

try:
    VERSION = re.sub(r'[\n\r\t"\']', "",subprocess.check_output(['git', 'describe', '--tags']).decode('utf-8'))
    DATE = re.sub(r'[\n\r\t"\']', "",subprocess.check_output(['git', 'log', '--pretty=format:"%ad"', '-1']).decode('utf-8'))
except:
    VERSION = '0.93'
    DATE = '2021-07-06'

#DEFAULT_CONFIG = 'config.json'



class GracefulKiller:
    """
    Catch the sigQUIT signal and handle it gracefully to initiate a proper shutdown of the modem
    """
    def __init__(self):
        self.kill_now = False
        # signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        signal.signal(signal.SIGQUIT, self.exit_gracefully)

    #@classmethod
    def exit_gracefully(self,signum, frame):
        print('killed')
        self.kill_now = True

def exitHandler():
    # the below code is useful for debugging hanging processes
    sys.stdout.flush()
 

        
atexit.register(exitHandler)

# parse arguments
parser = argparse.ArgumentParser(prog='py-CUDA-SDR', description = 'Software defined radio that performs Doppler search and supports multiple receiving stations and antennas')

parser.add_argument('-c', '--configFile', help='Configuration file to use', action='store', default = '')
parser.add_argument('-v',help=' -vv, -vvv increases verbosity',action='count',default=0)
parser.add_argument('-V','--version',help='Show program version', action = 'version',version = '%(prog)s ' + str(VERSION))
# parser.add_argument('--noGRC',help='does not start GNU Radio', action = 'count',default = 0)


args = parser.parse_args()

if args.configFile == '':
    print(f'Error: no config file specified')
    parser.print_help()
    sys.exit()

## logging
# use processlogger to access a proper multiprocess logging queue
logFmt = LOG_FORMAT
logLevel = max((1,30-(args.v * 10))) # increasing the number of v's increases the verbosity
# log.setLevel(logLevel) 
if args.v > 1:
	logFmt = LOG_FORMAT_DEBUG


logFileName = '{}_{}'.format(time.strftime("%Y_%m_%d_%H_%M", time.gmtime(time.time())), LOG_NAME)


# make a throwaway logger to log config loading output
tmp_log = logging.getLogger(LOG_NAME)
tmp_log.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler(sys.stdout)
tmp_log.addHandler(consoleHandler)
logFilePath = LOG_FOLDER + '/' + logFileName  + '.log'
fileHandler = logging.FileHandler(logFilePath)
fileHandler.setLevel(logging.DEBUG)
tmp_log.addHandler(fileHandler)


try:
    configFile = loadModularJson(args.configFile) # loads nested json config file
except Exception as e:
    tmp_log.error(f'Failed loading config file {args.configFile}')
    tmp_log.exception(e)
    tmp_log.error(f'Shutting down')

    raise e

# clean up the throwaway logger
while len(tmp_log.handlers) > 0:
    tmp_log.handlers[0].close()
    tmp_log.removeHandler(tmp_log.handlers[0])

del tmp_log


multiprocessing_logger.loggerSetup(LOG_NAME,logging.DEBUG,logFileName,LOG_FOLDER,logFmt,config=configFile)

log = logging.getLogger(LOG_NAME)

## Individual module log verbosity tuning
# demodulator_process.log.setLevel(logging.DEBUG)
# decoder_process.log.setLevel(logging.DEBUG)
modulator_process.log.setLevel(logging.INFO)
# rpcInterface.log.setLevel(logging.INFO)
softCombiner.log.setLevel(logging.INFO)

demodulator_process.log.setLevel(logLevel)
decoder_process.log.setLevel(logLevel)
# modulator_process.log.setLevel(logLevel)
rpcInterface.log.setLevel(logLevel)
# softCombiner.log.setLevel(logLevel)



# start
log.info("Starting py-cuda-sdr {} (commit date {})".format(VERSION,DATE))

log.info(f"Using config file {args.configFile}")



## initialize protocol -- TODO: change this
# first try to search for new-style multi protocols.
if "protocols" in configFile['Main'].keys():
    protocolNames = configFile['Main']['protocols'].keys()

    protocols = dict()
    protocolNamesDict = dict()
    for pName in protocolNames:
        protocolNamesDict[pName] = configFile['Main']['protocols'][pName]
        p = protocol.loadProtocol(protocolNamesDict[pName])
        protocols[pName] = p(conf=configFile)
        
    protocol = None # for backward compatibility
    log.info(f'Found protocols {protocolNamesDict}')
    
else:
    protocolName = configFile['Main']['protocol']
    protocolCls = protocol.loadProtocol(protocolName) # load the protocol

    #protocol.log.setLevel(logLevel) #### At the moment this is not working when passing the log to processes. ProtocolBase sets the level.  
    protocol = protocolCls(conf=configFile)
    log.info('Using protocol "{}"'.format(protocol.name))



radios = configFile['Radios']['Rx'].keys()
    

# Initialize modules
try:
    log.info('Utilizing {} channels: {}'.format(len(radios),', '.join([''.join(r) for r in radios])))

    tasks = []

    if not 'Tx' in configFile['Radios'].keys():
        log.warning('------- Modulator not intialized. Reason: \'Tx\' not found in config file -------')
        modulators = []
    else:
        modulators = []
        # If we have multiple protocols, we might have multiple modulators
        if protocol is None:
            modulatorNames = configFile['Radios']['Tx'].keys()
            for m in modulatorNames:
                modProtocol = configFile['Radios']['Tx'][m]['Protocol']
                modul = modulator_process.Modulator_process(configFile,protocols[modProtocol],m)
                modulators.append(modul)

        else:
            modul = modulator_process.Modulator_process(configFile,protocol)
            modulators.append(modul)

        tasks.extend(modulators)

    # check if we need the softCombiner -- should to be done before initializing the radios
    # check if the config is set to not use softCombiner
    if 'softCombiner_enabled' in configFile['Main'].keys():
        softCombinerEnabled = configFile['Main']['softCombiner_enabled']
    else:
        softCombinerEnabled = True

    if softCombinerEnabled and len(radios) > 1:
        # start the softCombiner
        combProc = softCombiner.SoftCombiner(configFile)
        tasks.append(combProc)
        log.info('SoftCombiner initialized.')
    else:
        # Change the port, such that the demodulator sends the data directly to the decoder
        configFile['Interfaces']['Internal']['decodeIn'] = configFile['Interfaces']['Internal']['demodIn']
        combProc = None
        log.info('SoftCombiner not started. Sending bytes directly to decoder')

    # start the radios
    demodulators = []
    for radio in radios:
        log.info(f'Initializing radio {radio}')
        if protocol is None:
            protocolName = configFile['Radios']['Rx'][radio]['Protocol']
            demodulators.append(demodulator_process.Demodulator_process(configFile,protocols[protocolName],radio))
        else:
            demodulators.append(demodulator_process.Demodulator_process(configFile,protocol,radio))

    tasks.extend(demodulators)


    # decoder
    if protocol is None: # multiple protocols
        dec = decoder_process.Decoder(configFile,protocols)
    else: # just one protocol
        dec = decoder_process.Decoder(configFile,protocol)
    tasks.append(dec)

    # RPC interface -- this is not added to the tasks list!
    rpcInt = rpcInterface.RpcInterface(configFile,modulators,demodulators,combProc)
except Exception as e:
    log.error('Fatal error while initializing. Message:')
    log.exception(e)
    sys.exit(-1)
    

startTime = time.time()   
        
# start processes

try:

    [t.start() for t in tasks]

    killer = GracefulKiller()

    while not killer.kill_now:

        # check whether all of the demodulators timed out while waiting for data from GRC
        demodTimeouts = 0
        for demodulator in demodulators:
            if demodulator.GRCTimeout() == True:
                demodTimeouts += 1
        if demodTimeouts == len(demodulators):
            break
            
        for t in tasks:
            time.sleep(.1)
            if not t.is_alive():
                raise Exception('Process %s died unexpectedly -- shutting down' %t.name)
       

except SystemExit as e:
    log.error('SYSTEMEXIT')
    print('system exit')
except Exception as e:
    log.error('Error in process')
    log.exception(e)

finally:
    log.info('Shutting down')

    # ask tasks to shut down
    log.info('Stopping processes')
    [t.stop() for t in tasks]
    log.info('Stop command sent to processes')

    try:
        plotData = dec.getVisualData()
    except Exception as e:
        log.error('could not generate plots')
        log.exception(e)
        
    # wait for tasks to shut down
    TIMEOUT = 5
    start = time.time()
    while time.time() - start <= TIMEOUT:
        if any(t.is_alive() for t in tasks):
            time.sleep(.1)
        else:
            # all processes are done
            break
    for t in tasks:
        log.info(f'{t.name} is alive? {t.is_alive()}')
        
    log.debug('Waiting for processes to shut down')
    for t in tasks:
        t.terminate()
        t.join()

    # process the data
    try:
        saveLoc = stats.processData(configFile,plotData,startTime,LOG_FOLDER)
    except Exception as e:
        log.error('Could not generate plots')
        log.exception(e)

    
    log.info('Finished -- Bye')

