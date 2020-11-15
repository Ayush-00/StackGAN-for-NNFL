## Separates the dataset into train and test in separate folders
import shutil
import pickle
import os


with open('train_test_split.txt') as f:
    train_test_split = f.read().splitlines()

with open('images.txt') as f:
    images = f.read().splitlines()
    # imageID, image name

#print(train_test_split)
train_filenames = []
test_filenames = []

#print(images)

count = 1;
for image in images:
    image = image.split()
    id = image[0]
    path = image[1]
    #print(path)
    if (int(train_test_split[int(id) - 1].split()[1]) == 1):
        #Value 1 = training, value 0 = test
        #print('Image ID = ' + str(id) + '\t Train or test = ' + str(train_test_split[int(id)].split()[1]))
        #print('Image ID = ' + str(id) + '\t Train')
        try:
            shutil.copyfile('./images/' +path, './train/' + path)
        except IOError as io_err:
            os.makedirs(os.path.dirname('./train/' + path))
            shutil.copyfile('./images/' +path, './train/' + path)

        train_filenames.append(path)
    else:
        #test
        #print('Image ID = ' + str(id) + '\t Train or test = ' + str(train_test_split[int(id)].split()[1]))
        #print('Image ID = ' + str(id) + '\t Test')
        try:
            shutil.copyfile('./images/' + path, './test/' + path)
        except IOError as io_err:
            os.makedirs(os.path.dirname('./test/' + path))
            shutil.copyfile('./images/' +path, './test/' + path)

        test_filenames.append(path)

    count+=1;
    if(count%100 == 0):
        print('Reached %sth image' % str(count))
    #if(count == 10):
    #   print('10 images moved. Exiting....')
    #   exit(0);

with open('train_filenames.pickle', 'wb') as f:
    pickle.dump(train_filenames, f)

with open('test_filenames.pickle', 'wb') as f:
    pickle.dump(test_filenames, f)
#
