# Loopback example using py-CUDA-SDR with GNU radio to interface to a Ettus USRP

This example runs the SDR with a TI CC11xx compatible framer. Data for packets is injected through ZMQ to the py-CUDA-SDR modulator. The modulator sends the data to [GNU radio](https://www.gnuradio.org/) which loops it through a Ettus USRP. The received data is sent to py-CUDA-SDR over ZMQ and is demodulated and decoded. The decoded data is sent to the python script and printed.

This example has been tested with the Ettus B210

## Dependencies:

- [GNU radio](https://www.gnuradio.org/)
- [Ettus UHD drivers](https://github.com/EttusResearch/uhd) 

## To run:

Start a terminal (terminal 1) and run

```
python uhd_graph.py
```

Start another terminal (terminal 2) and run

```
cd ../../py-cuda-sdr
python start_py-cuda-sdr.py -c ../examples/loopback_uhd/uhd_loopback_example.json -v
```

Start another terminal (terminal 3) and run

```
python create_and_receive_packets.py
```

Packets should now go through the SDR and be received.
Adjust the sliders in the GNU radio window to apply Doppler
Inspect the files for more information