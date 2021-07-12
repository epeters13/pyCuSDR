from __global__ import *
import matplotlib
try:
    matplotlib.use('pdf')
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import time
import logging
import pickle


log = logging.getLogger(LOG_NAME+'.'+__name__)
log.setLevel(logging.INFO)




def safeGet(key,dataCont):
    
    if key in dataCont.keys():
       return np.array(dataCont[key])
    else:
        
        log.warning('Key %s not found for worker %s' %(key,dataCont['workerId']))
        return np.array([])

def processData(conf,data,startTime = 0,log_folder = None):
    saveMsg = []

    if log_folder:
        # dump all the data in one file
        filetimeStamp = time.strftime("%Y_%m_%d_%H_%M_", time.gmtime(startTime))
        dataName = '%s/%sstat_data%s' %(log_folder,
                                            str(filetimeStamp),
                                            '.dat')
        with open(dataName,'wb') as f:
            pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    for d in data:
        saveMsg.append(processWorkerData(conf,d,startTime,log_folder))
    return saveMsg

def processWorkerData(conf,dataCont,startTime = 0, log_folder = None):

    workerId = safeGet('workerId',dataCont)
    doppler = safeGet('doppler',dataCont)
    doppler_std = safeGet('doppler_std',dataCont)
    timestamp = safeGet('timestamp',dataCont)
    spSymEst = safeGet('spSymEst',dataCont)
    SNR = safeGet('SNR',dataCont)
    numSyncSig = safeGet('numSyncSig',dataCont)
    packetSuc = safeGet('packetSuc',dataCont)
    packetFail = safeGet('packetFail',dataCont)
    numSlaves = safeGet('numSlaves',dataCont).astype(np.int)

    
    baudRate = safeGet('baudRate',dataCont)
    if len(baudRate) > 0:
        baudRate = baudRate[0]
        
    
    # We need to find the proper frequency for the radio.  If not found, use a default value (results in wrong rangerate estimates)
    keys = [k for k in conf['Radios']['Rx'].keys()]
    keyNames = [k.split('-')[0] for k in keys]
    Fc = None
    for i,k in enumerate(keyNames):
        if k in str(workerId):
            print('true')
            radioConf = conf['Radios']['Rx'][keys[i]]
            Fc = radioConf['frequency_Hz']
            samplesPerSlice = 2**conf['GPU'][radioConf['CUDA_settings']]['blockSize']-2**conf['GPU'][radioConf['CUDA_settings']]['overlap']
            log.info('Rx frequency for {} found as {}'.format(workerId,Fc))
            break
    if Fc == None:
        Fc = 1e8
        log.warning('No Rx frequency found for {}. {}'.format(workerId,Fc))
    

    try:
        rangerateMax = conf['Radios']['rangeRateMax']
    except Exception as e:
        log.warning('Could not find rangeRateMax in log, reason:\n{}'.format(e))
        rangerateMax = 12000

    log.debug('rangerateMax read to {}'.format(rangerateMax))
    
    spSym = np.mean(spSymEst)
    pkts = packetSuc + packetFail
    
    plt.ioff()                  # turn of interactive mode such that we can save the figure without showing
    Nplot = len(doppler)
    timeX = timestamp-startTime # makes all workers use the same reference time
    f = plt.figure()
    rangeRate = doppler*scipy.constants.speed_of_light/Fc
    ax1 = f.add_subplot(311)
    ax1.plot(timeX,rangeRate[:Nplot],label='Doppler')
    # ax1.plot(timeX,txFreqOffset[:Nplot],'--',label='Tx Freq offset') # estimated frequency offset
    tmp = np.where(pkts[:Nplot]>0)[0]
    tmpSuc = np.where(packetSuc[:Nplot]>0)[0]
    tmpErr = np.where(packetFail[:Nplot]>0)[0]

    packetNumVotes = numSlaves[tmp]
    colors = ['#800000','#ff0000','#ff6600','#ff9933','#ffcc66','#ffff66']
    # plot number of voters for each packet in a different colour
    if len(packetNumVotes) > 0:
        for i in range(int(np.max(packetNumVotes)+1)):
            tmpV = tmp[np.where(packetNumVotes==i)[0]]
            if len(tmpV) > 0:
                ax1.plot(timeX[tmpV],rangeRate[tmpV],'o',color=colors[i],label='packet {} votes'.format(i+1),markersize=2)

    ax1.plot(timeX[tmpSuc],rangeRate[tmpSuc],'kx',label='decoded',markersize=1)
    ax1.set_title('%s -- Doppler' %(workerId), fontsize='x-small')
    ax1.grid()
    ax1.legend(loc='lower center', mode='expand', fontsize='xx-small',ncol=4)
    ax1.set_ylabel('Range rate [m/s]', fontsize='x-small')
    # ax1.set_xlabel('time [s]')
    ax1.tick_params(axis = 'both', which = 'major',labelsize = 7)
    ax1.tick_params(axis = 'both', which = 'minor',labelsize = 5)

    rangerateStart = -1.35 * rangerateMax
    rangerateStop = 1.05 * rangerateMax
    ax1.set_ylim(rangerateStart,rangerateStop) # fix the limits of the plot

    ## This plot shows the number of sync sigs and voters
    ax3 = f.add_subplot(312,sharex=ax1)
    fixSyncsigAxis = False # hardcodes the axes in the doppler plot
    if 'Stats' in conf.keys():
        if 'FixSyncsigAxis' in conf['Stats'].keys():
            fixSyncsigAxis = conf['Stats']['FixSyncsigAxis']

    ax3.bar(timeX,numSyncSig[:Nplot])
    if fixSyncsigAxis:
        ax3.set_ylim([0,baudRate/8*1.1])
    ax3b = ax3.twinx()
    ax3b.plot(timeX,numSlaves+1,'.',color='red',markersize=1) # #003366
    ax3b.set_ylabel('number of votes', fontsize='x-small')
    ax3b.set_ylim([0,8])
    ax3b.tick_params(axis = 'both', which = 'major',labelsize = 7)
    ax3b.tick_params(axis = 'both', which = 'minor',labelsize = 5)
    # ax3.bar(tmp,numSyncH[tmp],'rx')
    ax3.grid()
    ax3.set_ylabel('Number of sync signals/second', fontsize='x-small')
    ax3.set_xlabel('time [s]', fontsize='x-small')
    ax3.tick_params(axis = 'both', which = 'major',labelsize = 7)
    ax3.tick_params(axis = 'both', which = 'minor',labelsize = 5)
    # ax1.set_xticklabels( () )

    ## Plot SNR
    ax4 = f.add_subplot(313,sharex=ax1)
    ax4.plot(timeX,SNR)
    ax4.set_ylim([-5,50])
    ax4.set_ylabel('SNR [dB]', fontsize='x-small')
    ax4.grid()
    ax4.tick_params(axis = 'both', which = 'major',labelsize = 7)
    ax4.tick_params(axis = 'both', which = 'minor',labelsize = 5)
  
    
    ax1.set_xlim(0,timeX[-1])



    
    if log_folder:

        figPrefix = str(time.strftime("%Y_%m_%d_%H_%M_", time.gmtime(startTime)))

        figName = '%s/%sdoppler_%s.pdf' %(log_folder,
                                          figPrefix,
                                          workerId)
        dataName = '%s/%sdoppler_%s' %(log_folder,
                                       figPrefix,
                                       workerId)
        
        f.set_size_inches(16.53,11.69) # Save as A3 
        # f.tight_layout()
        f.savefig(figName,bbox_inches='tight',format='pdf')
        plt.close(f)

        # save numpy data
        np.savez(dataName,
                 startTime = startTime,
                 timeX = timeX+startTime,
                 doppler = doppler[:Nplot],
                 rawDoppler = doppler[:Nplot],
                 doppler_std = doppler_std[:Nplot],
                 rangeRate =rangeRate[:Nplot],
                 packets = tmp,
                 packetsError = tmpErr,
                 packetsSuccess = tmpSuc,
                 numSyncSignals = numSyncSig[:Nplot],
        )

    log.info('Figure for worker %s saved as %s' %(workerId, figName ))
    return 'Figure for worker %s saved as %s' %(workerId, figName )


