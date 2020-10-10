
# coding: utf-8

# In[28]:


import json
import numpy as np
import os
import argparse


# In[29]:
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--gt_json', default='',
type=str, help='Trained state_dict file path to open')
parser.add_argument('--pred_json', default='result_test_1.json', type=str,
help='Dir to save results')
args = parser.parse_args()



with open(args.gt_json) as f:
    data = json.load(f)



classes = np.loadtxt('../labels.txt', delimiter=',', dtype=str)


# In[31]:


# print(classes)
class_dict = {}
for i in classes:
    class_dict[int(i[0])] = i[2]
    
print(class_dict.keys(), class_dict.values())


# In[32]:


ann = data['annotations']
# COCO_change_category = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 16, 21, 30, 33, 34, 51, 99]


# In[34]:


#create ground truth dictionary
gt = {}

for a in ann:
    if "bbox" in a.keys():
        image_name = a['image_id']
        cat_id = a['category_id']
        bbox = a['bbox']
        cat_name = class_dict[cat_id]
    #     score = a['score']

        bbox[2] = bbox[2] + bbox[0]
        bbox[3] = bbox[3] + bbox[1]
        temp = [cat_name]
        temp = temp + bbox

        if image_name in gt.keys():
            gt[image_name].append(temp)
        else:
            gt[image_name] = []
            gt[image_name].append(temp)


# In[35]:


#load predictions
with open('../eval/' + args.pred_json) as f:
    pred_ann = json.load(f)


# In[36]:


#create prediction dictionary
pred_dict = {}

for a in pred_ann:
    image_name = a['image_id']
    cat_id = a['category_id']
    bbox = a['bbox']
    cat_name = class_dict[cat_id]
    score = a['score']
    
#     bbox[2] = bbox[2] + bbox[0]
#     bbox[3] = bbox[3] + bbox[1]
    temp = [cat_name] + [score]
    temp = temp + bbox
        
    if image_name in pred_dict.keys():
        pred_dict[image_name].append(temp)
    else:
        pred_dict[image_name] = []
        pred_dict[image_name].append(temp)


# In[37]:


#eqauting the size
all_keys = gt.keys()

for key in all_keys:
    if key not in pred_dict.keys():
        print(key)
        pred_dict[key] = []


# In[38]:


all_pred_keys = list(pred_dict.keys())[:]
for key in all_pred_keys:
    if key not in gt.keys():
        del pred_dict[key]
        print(key)


# In[39]:


gt_dir_path = "ground-truth/"
pred_dir_path = 'predicted/'
# os.rmdir(pred_dir_path)
# os.rmdir(gt_dir_path)
os.mkdir(gt_dir_path)
os.mkdir(pred_dir_path)

#save ground truth annotations
for key in pred_dict.keys():
    f = open(gt_dir_path + key + '.txt', 'w')
    
    for l in gt[key]:
        f.write(' '.join([str(i) for i in l]) + '\n')
    f.close()

#save predicted annotations
for key in pred_dict.keys():
    f = open(pred_dir_path + key + '.txt', 'w')
    
    for l in pred_dict[key]:
        f.write(' '.join([str(i) for i in l]) + '\n')
    f.close()

