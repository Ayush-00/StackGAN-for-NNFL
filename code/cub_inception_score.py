import inception_score as in_score
import pickle
from model import STAGE1_G as G1
from model import STAGE2_G as G2
import torch
import numpy as np
from torch.autograd import Variable

def generate_imgs(embeddings_path, generator):
    #generator object: Stage1/2
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    generator.eval()
    num_imgs = len(embeddings)
    #scores = np.array(num_imgs)
    scores=[]
    batchsize = 10
    nz = 100
    for i in range(3):
        noise = Variable(torch.FloatTensor(batchsize, nz)).cuda()
        noise.data.normal_(0, 1)
        embedding = embeddings[i]
        input = torch.Tensor(embedding).cuda()
        #print(input.shape)
        _, fake_img, mu, logvar = generator(input, noise)
        #print(fake_img.shape)
        score, _ = in_score.inception_score(fake_img, batch_size=batchsize,resize = True)
        scores.append(score)
        print(str(i) + '/' + str(num_imgs) + ' done')
        del fake_img

    return scores

generator = G1()
path_pretrained = '../birds_stageI_2020_11_17_21_11_29/Model/netG_epoch_1.pth'
test_embeddings_path = '../data/cub/CUB_200_2011/test/char-CNN-RNN-embeddings.pickle'
generator.load_state_dict(torch.load(path_pretrained, map_location='cuda:0'))
generator.cuda()
scores = generate_imgs(test_embeddings_path,generator)

print(len(scores))
print(scores[0])
