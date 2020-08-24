
import torch


class TestDataset(torch.utils.data.Dataset):

    def __init__(self):

        self.data_list = []
        self.data_list2 = []
        for k in range(100):
            li = []
            li2 = []
            for kk in range(5):
                li.append( k*5+kk )
                li2.append( str(k*5+kk) )
                # str --> batch : list of tuple of str
                # int --> batch : list of Tensor of torch.int64
                # float --> batch : list of Tensor of torch.float64

            self.data_list.append( li )
            self.data_list2.append( li2 )

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, index):
        out = self.data_list[index]
        out2 = self.data_list2[index]
        #out = torch.tensor( out, dtype=torch.int16 )
        return out, out2


if __name__ == "__main__":

    td = TestDataset()
    dataloader = torch.utils.data.DataLoader(td, batch_size=2, shuffle=True, num_workers=1)

    for i_batch, (batch1, batch2) in enumerate(dataloader):
        #print(type(batch1), batch1[0], type(batch1[0]), batch1[0].dtype )
        #print(type(batch1[0]), type(batch2[0]))
        print(batch1)
        print(batch2)
        break
