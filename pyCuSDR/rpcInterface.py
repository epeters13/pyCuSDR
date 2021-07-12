# Copyright: (c) 2021, Edwin G. W. Peters

from __global__ import *
import threading
import logging
import socket

"""
RPC methods that can be accessed by higher level software to manage and monitor the SDR

"""

FC_TOL  = 1e5

log	= logging.getLogger(LOG_NAME+'.'+__name__)

from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler, Fault

# Restrict RPC requests to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

class RpcInterface(threading.Thread):

    quit = 0
    GRCRunning = False

    def __init__(self,conf,modulator=None,demodulator=None,softCombiner = None):
        threading.Thread.__init__(self)
        self.daemon = True
        self.conf = conf

        # if no demodulator and modulator set, use dummy class for unit testing
        if modulator is None or len(modulator) == 0:
            self.modulator = {'UHF':dummyModulator()}
        else:
            if not isinstance(modulator,list):
                self.modulator = {'UHF':modulator}
            elif len(modulator) > 0:
                # make a dictionary with modulators
                self.modulator = dict()
                for m in modulator:
                    self.modulator[m.name] = m

        if demodulator is None:
            self.demodulator = dummyModulator()
        else:
            if not isinstance(demodulator,list):
                self.demodulator = [demodulator]
            else:
                self.demodulator = demodulator

        self.softCombiner = softCombiner
                
        self._stopEvent = threading.Event()

        # Try to register GRC service control
        try:
            GRCServiceControlAddr = conf['Interfaces']['External']['GRCServiceControl']
            log.info('Registering GRC servicecontrol on %s' %(GRCServiceControlAddr))
            import xmlrpc.client
            self.GRCServiceControl = xmlrpc.client.Server(GRCServiceControlAddr)
        except Exception as e:
            log.error('Failed to register GRC service control. Reason: %s' %(e))

        socket.setdefaulttimeout(1)
        try:
            GRCRpcAddr = conf['Interfaces']['Internal']['GRCRpc']
            log.info('Registering GRC RPC interface on %s' %(GRCRpcAddr))
            import xmlrpc.client
            self.GRCRpc = xmlrpc.client.Server(GRCRpcAddr)
            self.GRCPpc.timeout = 1.0
        except Exception as e:
            log.error('Failed to register GRC interface control. Reason: %s' %(e))

                        
        rpcAddr = conf['Interfaces']['External']['XMLRpcIn']
        log.info('Registering XMLRPC server on %s' %(rpcAddr))
        addr,port = rpcAddr.rsplit(':',1)
        self.rpcServ = server = SimpleXMLRPCServer((addr,int(port)),
                                                   requestHandler=RequestHandler,
                                                   allow_none=True,
                                                   logRequests = False)
        server.register_introspection_functions() # allows system.listMethods, system.methodHelp and system.methodSignature

        
            
        # register GRC service control methods
        GRCMethods = [self.kill_GRC,
                      self.start_GRC]
        GRCMethodNames = ['kill',
                          'start']
        for m,n in zip(GRCMethods,GRCMethodNames):
            server.register_function(m,n)

        # register methods
        self.registerTxMethods(server)
        self.registerRxMethods(server)
        self.registerSTXMethods(server)
        self.registerSoftCombinerMethods(server)
        
        self.start()

        
        
    def registerTxMethods(self,server):

        @server.register_function
        def get_config():
            """Returns the config """
            return self.conf
            
        @server.register_function
        def get_Tx_rangerate(name = 'UHF'):
            """Returns the Tx rangerate [float]"""
            return self.modulator[name].rangerate

        @server.register_function
        def set_Tx_rangerate(rangerate):
            """Sets the Tx rangerate [float] for all modulators"""
            for modul in self.modulator.values():
                modul.rangerate = rangerate
            for demod in self.demodulator:
                # used by the demodulator to compute the frequency offset
                demod.TxRangerate = rangerate

        # sample frequency
        @server.register_function
        def get_Tx_samp_rate():
            """returns the Tx sample frequency [int]"""
            return  self.__get_GRC('get_tx_samp_rate')

        @server.register_function
        def get_STX_Tx_samp_rate():
            """returns the STX Tx sample frequency [int]"""
            return  self.__get_GRC('get_STX_Tx_sample_rate')

        #return self.modulator.Fs
        @server.register_function
        def set_Tx_samp_rate(Fs):
            """Set Tx sample rate in GRC -- note that this effectively modifies the data rate"""
            self.__set_GRC('set_tx_samp_rate',Fs)

        @server.register_function
        def set_STX_Tx_samp_rate(Fs):
            """Not implemented"""
            self.__set_GRC('set_STX_Tx_sample_rate',Fs)

            # raise NotImplementedError('Setting TxFs is not implemented')

        # centre frequency
        @server.register_function
        def get_Tx_freq():
            """returns the Tx centre frequency [int]"""
            try:
                return self.__get_GRC('get_Tx_Freq')
            except Fault as e:
                log.warning('Could not read GRC Rx frequency')
                for m in self.modulator:
                    if 'UHF' in self.name:
                        return m.Fc
                return 0
                

        @server.register_function
        def get_STX_Tx_freq():
            """returns the STX Tx centre frequency [int]"""
            try:
                return self.__get_GRC('get_STX_Tx_Freq')
            except Fault as e:
                log.warning('Could not read GRC Rx frequency')
                for m in self.modulator:
                    if 'STX' in self.name:
                        return m.Fc
                return 0

        @server.register_function
        def set_Tx_freq(Fc):
            """Not implemented"""
            raise NotImplementedError('Setting TxFc is not implemented')

        # centre frequency offset
        @server.register_function
        def get_Tx_freq_offset(name='UHF'):
            """ This config file value"""
            return self.modulator[name].centreFreqOffset

        @server.register_function
        def set_Tx_freq_offset(val,name='UHF'):
            """Read the fixed frequency offset for the radio. Used to adjust offsets compared to the receiver radio"""
            self.modulator[name].centreFreqOffset = int(val)

        @server.register_function
        def get_Tx_GRC_freq_offset(name='UHF'):
            """The frequency offset that is used to offset the DC level in GNU radio. This is added here and subtracted in GNU radio"""
            return self.modulator[name].freqOffset

        @server.register_function
        def get_Tx_total_freq_offset(name='UHF'):
            """Returns the centreFreqOffset + doppler. The actual frequency is then TxFc+TxtotalFreqOffset"""
            return self.modulator[name].totalFreqOffset

        @server.register_function            
        def get_Tx_gain():
            """Get the current Tx gain from GRC"""
            return self.__get_GRC('get_Tx_Gain')

        @server.register_function            
        def get_STX_Tx_gain():
            """Get the current Tx gain from GRC"""
            return self.__get_GRC('get_STX_Tx_Gain')

        @server.register_function            
        def set_Tx_gain(gain):
            """Set the Tx gain in GRC"""
            self.__set_GRC('set_Tx_Gain',gain)

        @server.register_function            
        def set_STX_Tx_gain(gain):
            """Set the STX Tx gain in GRC"""
            self.__set_GRC('set_STX_Tx_Gain',gain)

        @server.register_function            
        def get_Tx_baud_rate(name='UHF'):
            """Get the Tx baud rate"""
            return self.modulator[name].baudRate


        @server.register_function            
        def get_Tx_num_sync_flags(name='UHF'):
            """Returns the number of sync flags currently used"""
            return self.modulator[name].numSyncFlags
            
        @server.register_function            
        def set_Tx_num_sync_flags(val,name='UHF'):
            """Sets the number of sync flags currently used"""
            self.modulator[name].numSyncFlags = val

        
    def registerRxMethods(self,server):

        ## Rx methods
        # rangerate
        @server.register_function
        def get_Rx_rangerate(antenna=0):
            """Returns the Rx rangerate [float]"""
            return self.demodulator[antenna].rangerate

        
        @server.register_function
        def set_Rx_rangerate(rangerate,antenna=0):
            """Setting the Rx rangerate is not implemented"""
            raise NotImplementedError('Setting the RxRangeRate is not implemented')
            # warnings.warn('Setting the RxRangeRate is not implemented')

        @server.register_function            
        def get_Rx_baud_rate(antenna=0):
            """Baud rate provided by GRC"""
            return self.__get_GRC('get_baudRate')

        @server.register_function            
        def set_Rx_baud_rate(baud,antenna=0):
            """Baud rate provided by GRC"""
            self.GRCRpc.set_baud_rate(baud)
            
        @server.register_function            
        def get_Rx_baud_rate_est(antenna=0):
            """Baud rate estimated by modem"""
            return self.demodulator[antenna].baudRateEst

        # sample frequency
        @server.register_function            
        def get_Rx_samp_rate(antenna=0):
            """returns the Rx sample frequency [int]"""
            return self.GRCRpc.get_sample_rate()

        @server.register_function            
        def set_Rx_samp_rate(Fs,antenna=0):
            """Set the Rx sample rate in GRC -- Note that this does not alter the demodulator settings!"""
            try:
                self.GRCRpc.set_sample_rate(Fs)
                #self.demodulator[antenna].Fs 
            except Fault as e:
                Fs_read = self.GRCRpc.get_sample_rate()
                log.warning('Potential error setting STX sample rate. set to {} -- message {}'.format(Fs_read,e))
                if abs(Fs-Fs_read) > FS_TOL:
                    log.error('STX sample rate off: reading {} Hz, desired {} Hz'.format(Fs_read,Fs))
                    raise Exception('STX sample rate off: reading {} Hz, desired {} Hz'.format(Fs_read,Fs))

        # centre frequency
        @server.register_function            
        def get_Rx_freq(antenna=0):
            """returns the Rx IF centre frequency from Gnu radio. The RF centre frequency is then Rx_freq + Rx_GRC_freq_offset [int]"""
            # return self.demodulator.Fc
            try:
                return self.__get_GRC('get_Rx_freq')
            except Fault as e:
                log.warning('Could not read GRC Rx frequency')
                return self.demodulator[antenna].Fc

        @server.register_function            
        def set_Rx_freq(Fc,antenna=0):
            """Not implemented"""
            log.info('Setting Rx center frequency to {} Hz'.format(Fc))
            try:
                self.GRCRpc.set_Rx_Fc(Fc)
            except Fault as e:
                freq = self.__get_GRC('get_Rx_freq')
                log.warning('Setting GRC frequency might not have worked properly. Reading frequency {} Hz. Received error {}'.format(freq,e))
                if abs(freq - Fc) > FC_TOL:
                    log.error('Failed to set Rx frequency to {} Hz (read {} Hz) -- message {}'.format(Fc,freq,e))
                    raise Exception(e)
                #raise NotImplementedError('Setting Rx_freq is not implemented')


        @server.register_function            
        def get_Rx_GRC_freq_offset(antenna=0):
            """GRC offset from IF to RF centrefrequency"""
            return self.__get_GRC('get_Rx_Freq_Offset')

        @server.register_function            
        def get_Rx_freq_offset(antenna=0):
            """Returns the frequency offset where the modem locks on compared to Rx_freq. Effectively this value is Rx_GRC_freq_offset+Rx_IF_freq_offset_est+doppler"""
            return self.demodulator[antenna].RxFreqOffset

        @server.register_function            
        def get_Rx_IF_freq_offset_est(antenna=0):
            """ Frequency offset estimate from the RF centre frequency (Rx_freq + Rx_GRC_freq_offset). This is estimated by the modem based on the Tx range rate that is provided and the Rx range rate that is estimated."""
            return self.demodulator[antenna].RxIFFreqOffset

        @server.register_function            
        def set_Rx_IF_freq_offset_est(val,antenna=0):
            """Can't be set since it is an estimate computed in the modem"""
            return NotImplementedError('Setting Rx frequency offset is not implemented')

        @server.register_function            
        def get_Tx_IF_freq_offset_est(antenna=0):
            """ Tx IF frequency offset is estimated by the modem. This value is computed by the demodulator"""
            return self.demodulator[antenna].TxIFFreqOffset

        @server.register_function            
        def set_Tx_IF_freq_offset_est(val):
            return NotImplementedError('Setting Tx frequency offset is not implemented')

        @server.register_function            
        def get_Rx_SNR(antenna=0):
            return self.demodulator[antenna].SNR

        @server.register_function            
        def get_Rx_gain():
            """Returns the Rx gain set in GNU radio"""
            return self.__get_GRC('get_RxGain')

        @server.register_function            
        def set_Rx_gain(val):
            """Updates the Rx gain in GNU radio"""
            self.GRCRpc.set_RxGain(val)
            # self.__set_GRC('set_RxGain',val)

        @server.register_function
        def get_Rx_antenna_name(antenna=0):
            return self.demodulator[antenna].workerId

    def registerSTXMethods(self,server):

        ## S-band
        @server.register_function            
        def get_STX_gain():
            """Get the current STX rx gain from GRC"""
            return self.__get_GRC('get_STXGain')

        @server.register_function            
        def set_STX_gain(val):
            """Set the current STX rx gain from GRC"""
            self.GRCRpc.set_STXGain(val)

        @server.register_function            
        def get_STX_freq():
            """Get the current STX rx Fc from GRC"""
            return self.__get_GRC('get_STX_freq')

        @server.register_function            
        def set_STX_freq(Fc):
            """Get the current STX rx Fc from GRC"""
            log.info('Setting STX frequency to {} Hz')

            try:
                self.GRCRpc.set_STX_freq(Fc)
            except Fault as e:
                freq = self.GRCRpc.get_STX_freq()
                log.warning('Potential error setting STX sample rate. set to {} -- message {}'.format(Fs_read,e))
                
                if abs(freq-FC_TOL) > 1e6:
                    log.error('STX centre frequency reading {} Hz, desired {} Hz'.format(freq,Fc))
                    raise Exception('STX centre frequency reading {} Hz, desired {} Hz'.format(freq,Fc))
            
            
        @server.register_function            
        def get_STX_samp_rate():
            """Get the current STX rx sample rate from GRC"""
            return self.__get_GRC('get_STX_sample_rate')

        @server.register_function            
        def set_STX_samp_rate(Fs):
            """Set the current STX rx sample rate in GRC -- Note, that this does not alter the demodulator settings!"""
            try:
                self.GRCRpc.set_STX_sample_rate(Fs)
            except Fault as e:
                Fs_read = self.GRCRpc.get_STX_sample_rate()
                log.warning('Potential error setting STX sample rate. set to {} -- message {}'.format(Fs_read,e))
                if abs(Fs-Fs_read) > FS_TOL:
                    log.error('STX sample rate off: reading {} Hz, desired {} Hz'.format(Fs_read,Fs))
                    raise Exception('STX sample rate off: reading {} Hz, desired {} Hz'.format(Fs_read,Fs))

        @server.register_function            
        def get_STX_baud_rate():
            """Get the current STX baud rate"""
            return self.__get_GRC('get_STXbaudRate')

    def registerSoftCombinerMethods(self,server):
        ### SoftCombiner

        @server.register_function            
        def get_active_workers(timeout=0.25):
            """Get the names of the workers that submitted data since last poll"""
            if self.softCombiner:
                # return ['hello','bye']
                return self.softCombiner.getActiveWorkers(timeout)
            else:
                return NotImplementedError('Not available in single or client listen mode')
             
        
    def terminate(self):
        self._stopEvent.set()
        self.rpcServ.server_close()

        return 1
        
    def __del__(self):
        self.rpcServ.server_close()
        del self.rpcServ
        
    def run(self):

        while not self._stopEvent.is_set():
            self.rpcServ.handle_request()

        log.info('XMLRPC server terminated')
        

    ## Helper methods to interface GRC commands

    def __get_GRC(self,cmd):
        """
        Call a get RPC function to GRC.  Prints a nice error if GRC is not responding
        Input:
             cmd:      'get_method_name'
        """
        try:
            # It looks like: xmlrpclib.ServerProxy overrides getattr in a way that makes this work correctly
            return getattr(self.GRCRpc,cmd)()
        except Exception:
            msg = 'Could not execute GRC command \'{}\' -- Is GRC running?'.format(cmd)
            log.warning('RPC call error: {}'.format(msg))
            raise Exception(msg)

    def __set_GRC(self,cmd,val):
        """
        Call a set RPC function to GRC.  Prints a nice error if GRC is not responding
        Input:
             cmd:      'set_method_name'
             val:      value in appropriate data type
        """
        try:
            # It looks like: xmlrpclib.ServerProxy overrides getattr in a way that makes this work correctly
            getattr(self.GRCRpc,cmd)(val)
        except Exception:
            msg = 'Could not execute GRC command \'{}\' -- Is GRC running?'.format(cmd)
            log.warning('RPC call error: {}'.format(msg))
            raise Exception(msg)


    ## Methods to start and stop GRC on the service control RPC
    def start_GRC(self):
        """Used to forward the start() command to the GRC servicecontrol through RPC"""
        # try:
        log.info('Starting GRC')
        self.GRCServiceControl.start()
        self.GRCRunning = True
        return 1

    def kill_GRC(self):
        """Used to forward the kill() command to the GRC servicecontrol through RPC"""
        # try:
        log.info('Stopping GRC')
        self.GRCRunning = False
        try:
            socket.setdefaulttimeout(5) # make sure we have enough time to kill GRC
            self.GRCServiceControl.kill()
            socket.setdefaulttimeout(1) # back to 1 second
        except socket.timeout as e:
            raise TimeoutError('Timeout while attempting to kill GRC')
        return 1
    
class dummyModulator():

    _Fs = 256
    _rangerate = 10.2
    _Fc = 10000
    def __init__(self):
        """Implements modulator methods for unit testing"""
        pass

    @property
    def Fs(self):
        return self._Fs

    @property
    def Fc(self):
        return self._Fc

    @property
    def rangerate(self):
        return self._rangerate

    @rangerate.setter
    def rangerate(self,rangerate):
        log.warning('Dummy method received rangerate  %f'%(rangerate))
        self._rangerate = rangerate



if __name__ == "__main__":
    print('This method can not run stand alone. Unit test scripts are found in /test')
