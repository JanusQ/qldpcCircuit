from enumlikelydecoder import EnumDecoderCSC
from .decoder import Decoder
from BPdecoder import likelihoodDecoder
import numpy as np
class BPSampleDecoder(Decoder):
    def __init__(self,bp_iterations=100):
        name = "BPSample"+str(bp_iterations)
        super().__init__(name)
        self.name = name+"_Likelihood"
        self.bp_iterations = bp_iterations
        pass
    def decode(self,syndrome):
        return self.decoder.decode(syndrome)
    def set_h(self,chk,obs,priors):
        if not isinstance(chk,np.ndarray):
            chk = chk.toarray()
            obs = obs.toarray()
        self.decoder = likelihoodDecoder(chk,obs,priors,self.bp_iterations)

class EnumSampleDecoder(Decoder):
    def __init__(self,weight_num=4):
        name = "Enum"+str(weight_num)
        super().__init__(name)
        self.name = name+"_Likelihood"
        self.weight_num = weight_num
        pass
    def decode(self,syndrome):
        return self.decoder.decode(syndrome)
    def set_h(self,chk,obs,priors):
        if not isinstance(chk,np.ndarray):
            chk = chk.toarray()
            obs = obs.toarray()
        self.decoder = EnumDecoderCSC(chk,obs,priors,self.weight_num)
