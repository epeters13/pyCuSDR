# pyCuSDR
A GPU accelerated software defined radio with real-time Doppler search and correction. The Doppler search and demodulation is done using matched filters. This allows pyCuSDR to be adapted to any modulation scheme by supplying a different set of filters.

pyCuSDR features a correlation demodulator and decoder that triggers on sync and preamles. The decored can perform CRC and FEC. Check out [pyreedsolomon](https://github.com/epeters13/pyreedsolomon) for a fast Python Reed Solomon interface for Linux.

## Table of Contents

1. [Citing](#citing)
2. [Authors & Acknowledgments](#ack)
3. [Examples and how-tos](#bench)
4. [More information](#more)
5. [Licensing information](#license)

## Citing <a name="citing"></a>
If you use pyCuSDR in scholary publications, presentations or to communicate with your satellite, please cite the following work that presents the algorithms used:
```
    @ARTICLE{peters:2020:aismag,
        author={Peters, Edwin G. W. and Benson, Craig R.},
        journal={IEEE Aerospace and Electronic Systems Magazine}, 
        title={A Doppler Correcting Software Defined Radio Receiver Design for Satellite Communications}, 
        year={2020},
        volume={35},
        number={2},
        pages={38-48},
        doi={10.1109/MAES.2019.2960952}
    }
```

If you additionally want to cite this software package, please cite as follows:
```
    @MISC{pyCuSDR:repo,
        author={Edwin G. W. Peters},
        title={pyCuSDR: a GPU accelerated software defined radio with real time Doppler correction},
        month=Jul,
        year=2021,
        url={https://github.com/epeters13/pyCuSDR},
        note={\url{https://github.com/epeters13/pyCuSDR}}
    }
```

## Authors & Acknowledgments <a name="ack"></a>
This software was written and packaged by [Edwin G. W. Peters](https://github.com/mugpahug).

Special thanks to Craig R. Benson for proposing the idea and for the fruitful discussions

## Examples and running benchmarks <a name="bench"></a>

To run pyCuSDR start `pyCuSDR.py`

### Example using [GNU radio](https://www.gnuradio.org/) as the [UHD USRP](https://github.com/EttusResearch/uhd) interface

See README in [loopback_uhd_examples](examples/loopback_uhd_example/README.md) for instructions.

### Running bit error rate benchmarks

See README in [benchmark](examples/benchmark/README.md) for instructions.

## More information <a name="more"></a>

To support a different modulation scheme, check the `protocol` files in `pyCuSDR/protocol` and modulators under  `pyCuSDR/modulator/modulators`.
To support different encoders and decoders, check the `encoders` under `pyCuSDR/modulators/encoders` and `protocol` under `pyCuSDR/protocol`.

## License <a name="license"></a>
This software is licensed under the [BSD license](http://www.opensource.org/licenses/bsd-license.php). See the included [LICENSE](LICENSE) file for more information.
