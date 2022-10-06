'''Tests of basic file operations performed on originating catalog data'''

from dask.distributed import Client
import os
import hipscat as hc
import unittest

TEST_DIR = os.path.dirname(__file__)

class TestFileOps(unittest.TestCase):
    def test_blank(self):
        """Empty csv files exist, but do not contain columns to parse"""
        c = hc.Catalog('blank')
        input_path = os.path.join(TEST_DIR, 'data/blank/')
        # TODO - this should throw an error 
        # self.assertRaises(EmptyDataError, c.hips_import(file_source=input_path, fmt='csv'))
   
if __name__ == '__main__':
    unittest.main()
