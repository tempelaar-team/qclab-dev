from qc_lab import Data
from glob import glob
import sys 

pattern = str(sys.argv[1])

# Load all the data files
data_files = glob(pattern+'_*.h5')
data = Data()
for data_file in data_files:
    data_tmp = Data()
    data_tmp.load_from_h5(data_file)
    data.add_data(data_tmp)
data.save_as_h5(pattern+'.h5')
