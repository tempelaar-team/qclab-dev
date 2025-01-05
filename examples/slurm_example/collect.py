from qclab import Data
from glob import glob

# Load all the data files
data_files = glob('data_*.h5')
data = Data()
for data_file in data_files:
    data_tmp = Data()
    data_tmp.load_from_h5(data_file)
    data.add_data(data_tmp)
data.save_as_h5('data.h5')
