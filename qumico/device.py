from onnx.backend.base import DeviceType, Device


class QumicoDeviceType(DeviceType):
    OpenCL = DeviceType._Type(935) # arbitary num
    OpenMP = DeviceType._Type(936)
    ARMNeon =DeviceType._Type(937)


class QumicoDevice(Device):

    SUPPORT_DEVICE_TYPE=[];
    
    def __init__(self, device):
        super(QumicoDevice, self).__init__(device)
        self.options = []

    def _add_optinal_device_type(self, option):
        self.options.append(option)

    def is_valid_device_type(self):
        raise NotImplementedError()

    def get_compile_option(self, *args, **kwargs):
        raise NotImplementedError()

class RaspberryPi3(QumicoDevice):

    SUPPORT_DEVICE_TYPE=[QumicoDeviceType.ARMNeon,
                         QumicoDeviceType.OpenMP];

    PARALELL_SIZE_NEON = 4
    
    def __init__(self,neon=False, openmp=False, armv7a=True):
        super(RaspberryPi3, self).__init__("CPU")
        if neon:
            self._add_optinal_device_type(QumicoDeviceType.ARMNeon)

        if openmp:
            self._add_optinal_device_type(QumicoDeviceType.OpenMP)
        
        self.armv7a = armv7a
        

    def get_compile_option(self):
        ret = []
        if QumicoDeviceType.OpenMP in self.options:
            ret.append("-fopenmp")

        if QumicoDeviceType.ARMNeon in self.options:
            ret.append("-mfpu=neon")
            ret.append("-ftree-vectorize")

        if self.armv7a:
            ret.append("-march=armv7-a")
       
        return ret
