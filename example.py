from pprint import pprint

from loaders import ChronosDataset

group = 'm4_daily'

# the chronos train split actually also contains the test set (last 18 observations)
# ... at least for m4 monthly (didn't check others)
df, *_ = ChronosDataset.load_everything(group=group)
# df = ChronosDataset.load_data(group=group)

# ----- data set list

dt = ChronosDataset.get_chronos_datasets_names()
pprint(dt)
