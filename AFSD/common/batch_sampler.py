from torch.utils.data import Sampler
from AFSD.common.config import config


class BatchSampler(Sampler):

    def __init__(self, data_source, batch_size):
        super().__init__(data_source)

        self.batch_size = batch_size
        self.partition_point = config['training']['partition_num']
        self.normal_idxs = list(range(0, self.partition_point))
        self.abnormal_idxs = list(range(self.partition_point, len(data_source)))

    def __iter__(self):
        batch = []
        i = 0

        if self.batch_size == 1:
            while i < len(self.normal_idxs)+len(self.abnormal_idxs):
                batch.append(i)
                yield batch
                batch = []

        while i < len(self.abnormal_idxs):
            for b in range(int(self.batch_size/2)):
                batch.append(i)
                batch.append(i+self.partition_point)
                i += 1
            print(batch)
            yield batch
            batch = []

        # while i + self.batch_size < self.partition_point:
        #     for b in range(self.batch_size):
        #         batch.append(i)
        #         i += 1
        #     yield batch
        #     batch = []

    def __len__(self):
        return len(self.data_source)
