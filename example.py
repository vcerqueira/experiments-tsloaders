from pprint import pprint

from loaders import ChronosDataset

group = 'm5'
group = 'm5-RESAMPLE-MS-sum'

# the chronos train split actually also contains the test set (last 18 observations)
# ... at least for m4 monthly (didn't check others)
df, *_ = ChronosDataset.load_everything(group=group)
# df = ChronosDataset.load_data(group=group)

# import datasets
# ds = datasets.load_dataset(path='autogluon/chronos_datasets', name=group, split='train')
# ds.set_format("numpy")

# ----- data set list

dt = ChronosDataset.get_chronos_datasets_names()
pprint(dt)
