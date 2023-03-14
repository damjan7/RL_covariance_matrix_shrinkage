import numpy as np
import pandas as pd
import helper_functions as hf


class CovMatEstimation:

    def __init__(self, raw_data, start_date, end_date, pf_size):
        self.raw_data = raw_data
        self.start_date = start_date
        self.end_date = end_date
        self.pf_size = pf_size
