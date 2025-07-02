import qldpcdecoding
from qldpcdecoding.codes import gen_BB_code, gen_HP_ring_code
from qldpcdecoding.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoding.simulation.independentsim import independentnoise_simulation
from qldpcdecoding.simulation.bbcodesimmeas import circuit_level_simulation
from functools import reduce
import numpy as np
from rich.pretty import pprint
import ray
p = 0.005
css_code = gen_BB_code(72)
print(css_code.N)
print(css_code.hz.sum(axis=0))
bposddecoder = BPOSD_decoder()
bpdecoder = BP_decoder()
# reshapeddecoder = ReShapeBBDecoder(,css_code.lz,A,B)
# for windowsize in [2,3,4,5,6,7]:
#     print(f"windowsize={windowsize}")
pprint(circuit_level_simulation(css_code,p,[bposddecoder],num_trials=1000,num_repeat=6,method=0,plot=True,W=7))

