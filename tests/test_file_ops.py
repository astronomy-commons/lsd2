from dask.distributed import Client
#from hipscat import *
import hipscat as hc
import unittest

class TestFileOps(unittest.TestCase):
    def test_nonexistent(self):
        c = hc.Catalog('test_nonexistent')
   
if __name__ == '__main__':
    unittest.main()
