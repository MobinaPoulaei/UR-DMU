import torch
from options import *
from config import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def test(net, config, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("/home/user01/mojtaba_nafez/msad_dataset/gt_new.npy")
        frame_predict = None
        
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        for i in range(len(test_loader.dataset)):
            
            _data, _label, _name = next(load_iter)
            #print('==== test _data.shape before ===>', _data.shape)
            _data = _data.cuda().permute(0, 2, 1, 3)
            #print('==== test _data.shape after ===>', _data.shape)
            _label = _label.cuda()
            
            res = net(_data)
            a_predict = res["frame"]
            #print('test=======>', a_predict.shape) 1, 32
            temp_predict = torch.cat([temp_predict, a_predict], dim=1)
            # if (i + 1) % 10 == 0 :
            #     cls_label.append(int(_label))
            #     a_predict = temp_predict.mean(0).cpu().numpy()
            #     cls_pre.append(1 if a_predict.max()>0.5 else 0)          
            #     fpre_ = np.repeat(a_predict, 16)
            #     if frame_predict is None:         
            #         frame_predict = fpre_
            #     else:
            #         frame_predict = np.concatenate([frame_predict, fpre_])  
            #     temp_predict = torch.zeros((0)).cuda()
        temp_predict = temp_predict.cpu().detach().numpy()
        frame_predict = np.repeat(temp_predict, 16)
        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
    
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
        
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)

        # wind.plot_lines('roc_auc', auc_score)
        # wind.plot_lines('accuracy', accuracy)
        # wind.plot_lines('pr_auc', ap_score)
        # wind.lines('scores', frame_predict)
        # wind.lines('roc_curve',tpr,fpr)
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
        
