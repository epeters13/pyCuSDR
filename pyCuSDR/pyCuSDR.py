# Copyright: (c) 2021, Edwin G. W. Peters

from __global__ import *


import os
import sys
import time
import logging, logging.handlers

from pyLoadModularJson import loadModularJson
import multiprocessing_logger

import demodulator_process
import decoder_process
import softCombiner
import modulator_process
import rpcInterface
import stats
import protocol
import lib
from lib.gracefullKiller import GracefulKiller

LOG_FOLDER = '../logs'
if not os.path.exists(LOG_FOLDER):
    os.mkdir(LOG_FOLDER)

log = None

LOG_FORMAT	= '%(asctime)s.%(msecs)03d %(module)-16s -- %(message)s '
LOG_FORMAT_DEBUG	= '%(asctime)s.%(msecs)03d %(levelname)s %(module)-16s %(funcName)-22s:%(lineno)5s -- %(message)s'

try:
    VERSION = re.sub(r'[\n\r\t"\']', "",subprocess.check_output(['git', 'describe', '--tags']).decode('utf-8'))
    DATE = re.sub(r'[\n\r\t"\']', "",subprocess.check_output(['git', 'log', '--pretty=format:"%ad"', '-1']).decode('utf-8'))
except:
    VERSION = '0.93'
    DATE = '2021-07-06'

def exitHandler():
    # the below code is useful for debugging hanging processes
    sys.stdout.flush()

class PyCuSDR():
    """
    The main class that initializes and starts the pyCuSDR

    """

    def __init__(self,config_file_name,args):

        self.start_time = time.time() # used for file naming
        self.args = args
        # temporary logging to capture config file issues in the log
        TL = ThrowawayLogger(self.start_time)
        tmp_log = TL.get_log()
        
        # load config
        tmp_log.error(f'Loading configuration from {config_file_name}')
        try:
            self.configFile = loadModularJson(config_file_name) # loads nested json config file
        except Exception as e:
            tmp_log.error(f'Failed loading config file {args.configFile}')
            tmp_log.exception(e)
            tmp_log.error(f'Shutting down')

            raise e

        # now we can kill the temporary log
        del TL
        # setup the proper pass log
        self._init_logging()
        global log
        log = self.log
        # start
        log.info("Starting pyCu-SDR {} (commit date {})".format(VERSION,DATE))

        log.info(f"Using config file {args.configFile}")

        # load the protocols
        self._init_protocols() # load the protocols
        self.radios = self.configFile['Radios']['Rx'].keys()

        # intialize the SDR modules
        self._init_SDR()

    def terminate(self):
        """
        Clean termination of all the processes and processing of stats data
        """
        self._shutdown()

        
    def is_alive(self):
        """
        Check if all the SDR processes are alive
        """
        if self.killer.kill_now:
            log.debug('killed')
            return False
        else:
            # check whether all of the demodulators timed out while waiting for data from GRC
            try:
                demodTimeouts = 0 # keep running until all demodulators have nothing to do
                for demodulator in self._demodulators:
                    if demodulator.GRCTimeout() == True:
                        demodTimeouts += 1
                if demodTimeouts == len(self._demodulators):
                    log.debug('all demodulators timed out')
                    return False

                for t in self.tasks:
                    time.sleep(.1)
                    if not t.is_alive():
                        raise Exception('Process %s died unexpectedly -- shutting down' %t.name)
                    
            except SystemExit as e:
                log.error('SYSTEMEXIT')
            except Exception as e:
                log.error('Error in process')
                log.exception(e)
                raise Exception("pyCuSDR terminated unexpectedly") from e

        return True

    def start(self):
        """
        Start the pyCuSDR tasks
        Use is_alive() to check the state
        Use terminate() to perform a clean shutdown
        """
        self._running = True
        [t.start() for t in self.tasks]        
        self.killer = GracefulKiller()

        
    def run(self):
        """
        Start pyCuSDR and loop until terminated
        """
        try:
            self.start()
            while self.is_alive():
                time.sleep(0.1)
        except SystemExit as e:
            log.error('SYSTEMEXIT')
        except Exception as e:
            log.error('Error in process')
            log.exception(e)
        finally:
            self.terminate()


    def shutdown_tasks(self,*args,**kwargs):
        """
        Overridable. Called in _shutdown. Allows extra tasts to be performed prior to terminating all processes
        """
        self._running = False
        try:
            log.info('Saving pass data and generating plots')
            plotData = self.dec.getVisualData()
            saveLoc = stats.processData(self.configFile,plotData,self.start_time,LOG_FOLDER)
            log.info('Finished saving plots and data')
        except Exception as e:
            log.error('could not generate plots')
            log.exception(e)
        
    def _shutdown(self):
        # internally called upon termination of the SDR. fetches and accumulates stats and terminates the tasks
        log.info('Shutting down')

        # ask tasks plolitely to shut down
        log.info('Stopping processes')
        [t.stop() for t in self.tasks]
        log.info('Stop command sent to processes')

        self.shutdown_tasks()
        # kill the rpc server
        self.rpcInt.terminate()

        # wait for tasks to shut down
        TIMEOUT = 5
        start = time.time()
        while time.time() - start <= TIMEOUT:
            if any(t.is_alive() for t in self.tasks):
                time.sleep(.1)
            else:
                # all processes are done
                break
        for t in self.tasks:
            log.debug(f'{t.name} is alive? {t.is_alive()}')

        # ask tasks slightly less polite to shut down and finish
        for t in self.tasks:
            t.terminate()
            t.join()

        self.rpcInt.join()
        
        log.info('Finished -- Bye')

            
    def _init_SDR(self):
        # intialize all the SDR modules that the config file asks us to
        try:
            log.info(f'Utilizing {len(self.radios)} channels: {", ".join(["".join(r) for r in self.radios])}')

            self.tasks = tasks = []

            # initialize modulators
            if not 'Tx' in self.configFile['Radios'].keys():
                log.warning('------- Modulator not intialized. Reason: \'Tx\' not found in config file -------')
                self._modulators = []                
            else:
                self._modulators = []
                # If we have multiple protocols, we might have multiple modulators
                modulatorNames = self.configFile['Radios']['Tx'].keys()
                for m in modulatorNames:
                    modProtocol = self.configFile['Radios']['Tx'][m]['Protocol']
                    modul = modulator_process.Modulator_process(self.configFile,self.protocols[modProtocol],m)
                    self._modulators.append(modul)
                    

                self.tasks.extend(self.modulators)
         
            # check if we need the softCombiner -- should to be done before initializing the radios
            # check if the config is set to not use softCombiner
            if 'softCombiner_enabled' in self.configFile['Main'].keys():
                softCombinerEnabled = self.configFile['Main']['softCombiner_enabled']
            else:
                softCombinerEnabled = False

            if softCombinerEnabled and len(self.radios) > 1:
                # start the softCombiner
                combProc = softCombiner.SoftCombiner(self.configFile)
                tasks.append(combProc)
                log.info('SoftCombiner initialized.')
            else:
                # Change the port, such that the demodulator sends the data directly to the decoder
                self.configFile['Interfaces']['Internal']['decodeIn'] = self.configFile['Interfaces']['Internal']['demodIn']
                combProc = None
                log.info('SoftCombiner not started. Sending bytes directly to decoder')

            # start the receivers
            self._demodulators = []
            for radio in self.radios:
                log.info(f'Initializing radio {radio}')
                protocolName = self.configFile['Radios']['Rx'][radio]['Protocol']
                self._demodulators.append(demodulator_process.Demodulator_process(self.configFile,self.protocols[protocolName],radio))

            tasks.extend(self._demodulators)

            # start the decoder. Keep link to class since we use this on termination to fetch data
            self.dec = decoder_process.Decoder(self.configFile,self.protocols)
            tasks.append(self.dec)

            self.rpcInt = rpcInterface.RpcInterface(self.configFile,self._modulators,self._demodulators,combProc)

        except Exception as e:
            log.error('Fatal error while initializing. Message:')
            log.exception(e)
            sys.exit(-1)

    
    def _init_logging(self):
        # start the proper log
        logLevel = max((1,30-(self.args.v * 10))) # increasing the number of v's increasing the number of v's increases the verbosity
        if self.args.v > 1:
	        logFmt = LOG_FORMAT_DEBUG
        else:
            logFmt = LOG_FORMAT
        logFileName = '{}_{}'.format(time.strftime("%Y_%m_%d_%H_%M", time.gmtime(self.start_time)), LOG_NAME)

        multiprocessing_logger.loggerSetup(LOG_NAME,logLevel,logFileName,LOG_FOLDER,logFmt,config=self.configFile)
        
        self.log = logging.getLogger(LOG_NAME)
        # set the log level for all the sub processes
        demodulator_process.log.setLevel(logLevel)
        decoder_process.log.setLevel(logLevel)
        modulator_process.log.setLevel(logLevel)
        rpcInterface.log.setLevel(logLevel)
        softCombiner.log.setLevel(logLevel)

        
    def _init_protocols(self):
        
        if "protocols" in self.configFile['Main'].keys():
            protocolNames = self.configFile['Main']['protocols'].keys()

            self.protocols = dict()
            self.protocolNamesDict = dict()
            for pName in protocolNames:
                self.protocolNamesDict[pName] = self.configFile['Main']['protocols'][pName]
                p = protocol.loadProtocol(self.protocolNamesDict[pName])
                self.protocols[pName] = p(conf=self.configFile)

            log.info(f'Found protocols {self.protocolNamesDict}')

        else:
            raise KeyError('"protocols" not defined in "Main". Check template config files for howto\'s')
            

    ## getters

    @property
    def demodulators(self):
        return self._demodulators

    @property
    def modulators(self):
        return self._modulators

        
class ThrowawayLogger(object):
    """
    This creates a simple log that catches any errors that occur while loading the configuration file
    """
    def __init__(self,start_time=time.time()):
        logFileName = '{}_{}'.format(time.strftime("%Y_%m_%d_%H_%M", time.gmtime(start_time)), LOG_NAME)

        # make a throwaway logger to log config loading output
        self.tmp_log= tmp_log  = logging.getLogger(LOG_NAME)
        tmp_log.setLevel(logging.DEBUG)
        consoleHandler = logging.StreamHandler(sys.stdout)
        tmp_log.addHandler(consoleHandler)
        logFilePath = LOG_FOLDER + '/' + logFileName  + '.log'
        fileHandler = logging.FileHandler(logFilePath)
        fileHandler.setLevel(logging.DEBUG)
        tmp_log.addHandler(fileHandler)

        
    def get_log(self):
        # return the log handler
        return self.tmp_log

    
    def __del__(self):
        while len(self.tmp_log.handlers) > 0:
            self.tmp_log.handlers[0].close()
            self.tmp_log.removeHandler(self.tmp_log.handlers[0])
    
        del self.tmp_log


        
    


if __name__ == "__main__":

    import argparse
    import atexit

    atexit.register(exitHandler)


    # parse arguments
    parser = argparse.ArgumentParser(prog='pyCu-SDR', description = 'Software defined radio that performs Doppler search and supports multiple receiving stations and antennas')

    parser.add_argument('-c', '--configFile', help='Configuration file to use', action='store', default = '')
    parser.add_argument('-v',help=' -vv, -vvv increases verbosity',action='count',default=0)
    parser.add_argument('-V','--version',help='Show program version', action = 'version',version = '%(prog)s ' + str(VERSION))
    # parser.add_argument('--noGRC',help='does not start GNU Radio', action = 'count',default = 0)


    args = parser.parse_args()

    if args.configFile == '':
        print(f'Error: no config file specified')
        parser.print_help()
        sys.exit()

    sdr = PyCuSDR(args.configFile,args)

    ## this keeps the SDR in a forever loop
    # sdr.run()

    ## here we can fine grain the control in the main loop
    sdr.start()

    try:
        while sdr.is_alive():
            time.sleep(0.1)
    except:
        pass
    finally:
        # Proper shutdown and data generation
        sdr.terminate()
        
