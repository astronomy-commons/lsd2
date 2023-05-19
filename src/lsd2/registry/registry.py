import os

from .. import hipscat as hc

class Registry:

    def __init__(self, location: str = '', list_available=True):

        self.location = location
        self.catalogs = {}

        if not os.path.exists(location):
            raise Exception(f"Path to hipscat dir: '{location}' does not exist")
        
        subdirs = [x for x in os.listdir(location) if os.path.isdir(os.path.join(location, x))]
        for sub in subdirs:
            try:
                cat = hc.Catalog(sub, location)
                self.catalogs[sub] = cat
            except:
                pass

        if len(self.catalogs) == 0:
            raise Exception(f"No lsd2.hipcat.Catalogs in dir: '{location}'")

        if list_available:
            self.list_hipscats()


    def list_hipscats(self):
        print("Available HiPSCats:")
        for cat in self.catalogs:
            print(f"   {cat}")


    def load_hipscat(self, catname):
        if catname in self.catalogs.keys():
            return self.catalogs[catname]
        else:
            raise Exception(f"Catalog: '{catname}' not found in registry")
