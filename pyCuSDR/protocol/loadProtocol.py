# Copyright: (c) 2021, Edwin G. W. Peters

def loadProtocol(protocolName):
    # add proper radio protocols here
    if protocolName == 'CC11xx':
        from protocol.CC11xx import CC11xx as protocolCls

    # benchmark protocols
    elif protocolName == 'bench_GMSK':
        from protocol.benchmark.bench_GMSK import Bench_GMSK as protocolCls
    elif protocolName == 'bench_BPSK':
        from protocol.benchmark.bench_BPSK import Bench_BPSK as protocolCls
    elif protocolName == 'bench_FSK':
        from protocol.benchmark.bench_FSK import Bench_FSK as protocolCls
    elif protocolName == 'bench_GFSK':
        from protocol.benchmark.bench_GFSK import Bench_GFSK as protocolCls
    else:
        raise ImportError('Protocol %s does not exist' %(protocolName))

    return protocolCls

