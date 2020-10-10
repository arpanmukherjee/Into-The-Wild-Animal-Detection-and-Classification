# from future import print_function
import sys
import os
import cv2
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import COCO_ROOT, COCO_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import COCO_CLASSES, COCOAnnotationTransform, COCODetection
import torch.utils.data as data
from ssd import build_ssd

COCO_change_category = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 16, 21, 30, 33, 34, 51, 99]
COCO_change_category = [str(i) for i in COCO_change_category]

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='ssd300_SE_BN_2_2000.pth',
type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.25, type=float,
help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
help='Use cuda to train model')
parser.add_argument('--test_folder', default="./validation", help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
# dump predictions and assoc. ground truth to text file for now
    filename = save_folder + 'result.json'
    num_images = len(testset)
    Final_list = []
    path = args.test_folder

    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        # img = testset.pull_image(i)
        img = cv2.imread(path + "/" + testset[i], cv2.IMREAD_COLOR)

        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()


        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

        # ii -> category id
        for ii in range(detections.size(1)):
            j = 0
            while detections[0, ii, j, 0] >= thresh:

                score = detections[0, ii, j, 0].cpu().data.numpy()
                pt = (detections[0, ii, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])

                # standard format of coco ->
                # [{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236},{...},...]
                temp_dict = {}

                img_name = testset[i].split(".")
                img_id = img_name[0]
                # temp_dict["image_id"] = str(testset.pull_anno(i)[0]['image_id'])
                temp_dict["image_id"] = img_id
                temp_dict["category_id"] = float(COCO_change_category[ii])
                temp_dict["bbox"] = [float(c) for c in coords]
                temp_dict["score"] = float(score)

                Final_list.append(temp_dict)

                # with open(filename, mode='a') as f:
                #     f.write(
                #         '{"image_id":' + str(testset.pull_anno(i)[0]['image_id']) +
                #         ',"category_id":' + str(COCO_change_category[ii]) +
                #         ',"bbox":[' + ','.join(str(c) for c in coords) + ']'
                #         ',"score":')
                #     f.write('%.2f' %(score))
                #     f.write('},')
                    # you need to delete the last ',' of the last image output of test image
                j += 1
    import json
    with open(filename, 'w') as f:
        json.dump(Final_list, f)


def test_voc(path):
# load net
    print(args.trained_model)
    num_classes = 17 # change
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load("weights/" + args.trained_model))
    net.eval()
    print('Finished loading model!')



    # load data
    # testset = COCODetection(args.coco_root, None, COCOAnnotationTransform)
    testset = os.listdir(path)
    print(testset)


    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
    BaseTransform(net.size, (104, 117, 123)),
    thresh=args.visual_threshold)

if __name__ == '__main__':
    test_path = args.test_folder
    test_voc(test_path)