"""
A script to load the active config and logger for tests
"""


import json, rjsmin
import logging
import os,sys
import time
sys.path.append('../src')
from protocol import loadProtocol

def getConfigAndLog(configFilePath,logLevel = logging.DEBUG):

        
    LOG_NAME = 'TEST'
    LOG_FORMAT          = '%(asctime)s -- %(message)s'
    LOG_FORMAT_DEBUG    = '%(asctime)s %(module)-16s %(funcName)-22s:%(lineno)5s -- %(message)s'


    currentPath = os.path.realpath(__file__).strip(os.path.basename(__file__))

    if logLevel:
        log = logging.getLogger(LOG_NAME)

        # logFmt = LOG_FORMAT_DEBUG
        log.setLevel(logLevel)
        logging.Formatter.converter = time.gmtime
        # logFormatter = logging.Formatter(logFmt,  "%Y-%m-%d %H:%M:%S")

        consoleHandler = logging.StreamHandler(sys.stdout)
        # consoleHandler.setFormatter(logFormatter)
        log.addHandler(consoleHandler)

        

    with open(configFilePath, mode="r") as cfgFile:
        strippedJSONFile = rjsmin.jsmin(cfgFile.read())
        configFile = json.loads(strippedJSONFile)
    # except FileNotFoundError as e:
    #     log.error('Config file %s not found. Template can be found in conf.json.default' %(args.configFile))    
    #     raise FileNotFoundError('Config file %s not found. Template can be found in conf.json.default' %(args.configFile)) from e
    
    return configFile
    
