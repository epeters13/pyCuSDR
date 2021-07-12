# Bit error rate benchmarks through pyCuSDR

These modules are made to directly benchmark the bit error rate (BER) performance of pyCuSDR for a few select modulation schemes. The benchmark scripts applies noise to the data and injects it straight into the pyCuSDR demodulator. The BER performance of the received data is reported.

## Getting more help

Run `python bench_sdr.py` to print the instructions

## Running a benchmark

Start terminal #1
```
cd pyCuSDR
python pyCuSDR.py -c ../config/benchmark/bench_XXXX.json -v
```
where XXXX is the modulation scheme that is desired

Start terminal #2
```
cd test/benchmarks
python bench_sdr.py XXXX N
```
where XXXX is the matching modulation scheme and N is the number of tests
