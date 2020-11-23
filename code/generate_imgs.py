import inception_score as in_score
import pickle
from model import STAGE1_G as G1
from model import STAGE2_G as G2
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.utils.data as data

dest_folder_path = '../output/generated_images/'
path_pretrained = '../birds_stageI_2020_11_17_21_11_29/Model/netG_epoch_1.pth'
test_embeddings_path = '../data/cub/CUB_200_2011/test/char-CNN-RNN-embeddings.pickle'

def generate_fake_imgs(generator, dataloader):
    scores = []
    #batchsize = dataloader[0].shape[0]
    batchsize = 16

    for batch_no, embedding in enumerate(dataloader):
        noise = Variable(torch.FloatTensor(batchsize, nz)).cuda()
        noise.data.normal_(0, 1)
        input = torch.Tensor(embedding).cuda()
        _, fake_img, mu, logvar = generator(input, noise)
        #print(fake_img.shape)
        for img_no in range(batchsize):
            save_image(fake_img[img_no], dest_folder_path + str(batch_no) + '_' + str(img_no) + '.jpg')
        #score, _ = in_score.inception_score(fake_img, batch_size=batchsize//2,resize = True)
        #print('Batch number %d score calculated..' % i)
        #print(score)
        #scores.append(score)
        if(batch_no == 3):
            exit(0)
        del fake_img



class TestDataset(data.Dataset):
    def __init__(self, embeddings_path):
        with open(embeddings_path, 'rb') as f:
            #Shape (2933, 10, 1024). 10 is number of sentences for an image
            self.embeddings_acc_to_img = pickle.load(f)
        print("Embeddings loaded!")
        self.embeddings = np.array(self.embeddings_acc_to_img).reshape((-1, 1024))

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, index):
        return self.embeddings[index]



if __name__ == "__main__":
    batch_size = 16
    nz = 100

    generator = G1()

    generator.load_state_dict(torch.load(path_pretrained, map_location='cuda:0'))
    generator.cuda()
    dataset = TestDataset(test_embeddings_path)
    dataloader = data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    generate_fake_imgs(generator, dataloader)
