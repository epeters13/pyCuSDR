# Copyright: (c) 2021, Edwin G. W. Peters

import multiprocessing_logging

import time
import logging
import sys, os
import socket


def loggerSetup(logName,logLevel,logFileName,logFolder,logFmt,config = None, logLevelConsole = None):
    """
    Configure the logger for pyCuSDR
    All provided arguments override the config setting when specified

    set logLevel to None if the level has to be read from config
    """

    if logLevel is None:
        lvl = config['Logging']['log_level']
        if lvl == "DEBUG":
            logLevel = logging.DEBUG
        elif lvl == "INFO":
            logLevel = logging.INFO
        elif lvl == "WARNING":
            logLevel = logging.WARNING
        elif lvl == "ERROR":
            logLevel = logging.ERROR

            
    if not logLevelConsole:
        logLevelConsole = logLevel

    log = logging.getLogger(logName)

    # setup the log
    log.setLevel(logLevel)

    logging.Formatter.converter = time.gmtime
    consoleHandler = logging.StreamHandler(sys.stdout)
        
    consoleHandler.setLevel(logLevelConsole)
    log.addHandler(consoleHandler)

    if logFmt:
        logFormatter = logging.Formatter(logFmt,  "%Y-%m-%d %H:%M:%S")
        consoleHandler.setFormatter(logFormatter)

    if not os.path.exists(logFolder):
        os.makedirs(logFolder)

    logFilePath = logFolder + '/' + logFileName  + '.log'
    fileHandler = logging.FileHandler(logFilePath)
    if logFmt:
        fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)

    memoryHandler = logging.handlers.MemoryHandler(20,flushLevel=logging.ERROR,target=fileHandler,flushOnClose=True)
    log.addHandler(memoryHandler)

            
    # This magic makes logging from multiple processes possible
    multiprocessing_logging.install_mp_handler() 

    log.info('Logging to %s' %(logFilePath))
    
    return log
