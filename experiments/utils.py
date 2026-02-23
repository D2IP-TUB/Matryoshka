import cpuinfo
import os
import platform
import psutil
import re
import polars as pl


def get_system_info():
    info = {}
    info['platform'] = platform.system()
    info['architecture'] = platform.machine()
    info['processor'] = cpuinfo.get_cpu_info()['brand_raw']
    info['ram'] = f'{round(psutil.virtual_memory().total / (1024.0 **3))} GB'
    
    return info