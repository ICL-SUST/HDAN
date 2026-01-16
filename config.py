from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import os

# --------------------------- Configuration Object ---------------------------
C = edict()
config = C
cfg = C

# ----------------------------------Common Settings----------------------------
C.seed = 12345

# --------------------------- Network & Optimization ---------------------------
C.pre_trained = True
C.num_epoch = 500
C.lr_S = 2e-4
C.lr_D = 2e-5
C.momentum_S = 0.9
C.momentum_D = 0.9
C.step_size_S = 5000
C.step_size_D = 5000
C.beta1 = 0.9
C.beta2 = 0.999
C.batch_train = 4

# --------------------------- CUDNN ---------------------------
cudnn.enabled = True
cudnn.benchmark = True

# --------------------------- Data setting ---------------------------
C.traindata_path = './PretermInfant_dataset/iSeg2019_hdf5/iSeg-2019-Training/'
C.valdata_path = './PretermInfant_dataset/iSeg2019_hdf5/iSeg-2019-Validation/'

# --------------------------- Data & Model Params --------------------------
C.input_dim = 2
C.ignore_label = 9
C.num_classes = 4
C.crop_size = (64, 64, 64)

# --------------------------- checkpoint & Note ---------------------------
C.checkpoint_name = 'InfantSeg'
C.note_S = '3dbrainseg(Adam lr_S: ' + str(C.lr_S) + ',w_decay:1e-4' + 'beta:' + str(C.beta1) + ',' + str(C.beta2) + ',' + 'step:' + str(C.step_size_S) + ' , lr_step)'
C.note_D = '3dbrainseg(Adam lr_S: ' + str(C.lr_S) + ',w_decay:1e-4' + 'beta:' + str(C.beta1) + ',' + str(C.beta2) + ',' + 'step:' + str(C.step_size_S) + ' , lr_step)'
C.num_checkpoint = '20000'
C.note = str(C.num_checkpoint) + '_' + C.checkpoint_name

# --------------------------- Testing ---------------------------
C.checkpoint = './checkpoints/'+str(C.num_checkpoint) + '_' + C.checkpoint_name + '.pth'

# --------------------------- Validation ---------------------------
print('@%s:  ' % os.path.basename(__file__))
print("well done")