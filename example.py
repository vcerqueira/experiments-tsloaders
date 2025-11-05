from loaders import ChronosDataset

group = 'm4_quarterly'

train, *_ = ChronosDataset.load_everything(group=group, split='train')
# test, *_ = ChronosDataset.load_everything(group=group, split='test')


