CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#CFLAGS = -O -m pdb
#DEBUG = --debug

#the regex of the slices in the target dataset
#for the prostate
RGX = Case
G_RGX = Case\d+_\d+

SAUG_DATA_NORM = [('IMGaug', nii_transform_normalize, False), ('GTaug', nii_gt_transform, False), ('GTaug', nii_gt_transform, False)]
T_DATA = [('IMG', nii_transform, False), ('GTNew', nii_gt_transform, False), ('GTNew', nii_gt_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[0.1,0.9],'moment_fn':'soft_size',}, None, None, None, 1),]
NET = UNet

LSIZE = [('EntKLProp', {'moment_fn':'soft_size','lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

LSIZEQuadDist = [('EntKLPropWMoment', {'lamb_moment':0.0001,'matrix':False, 'rel_diff':False,'temp':1.01,'margin':0.1,'mom_est':[[112.33, 21.51],[112.17, 18.08]],'moment_fn':'soft_dist_centroid','lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]

LSIZECentroid = [('EntKLPropWMoment', {'weights_se':[0.1,0.9],'lamb_se':1,'lamb_moment':0.0001,'rel_diff':False, 'matrix':False, 'temp':1.01,'margin':0,'mom_est':[[191.52, 192.12],[191.55, 188.53]],'moment_fn':'soft_centroid', 'lamb_consprior':1,'ivd':True,'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]


# the folder containing the target dataset - site A is the target dataset and site B is the source one
T_FOLD = ./SAML/data/SA/

# run the TTA with size only for 150 epc
M_WEIGHTS_ul = results/prostate/enklsize/

#run the main experiment
TRN = results/prostate/enklsizecent results/prostate/enklsizedist

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-CSize.tar.gz

all: pack
plot: $(PLT)

pack: $(PACK) report
$(PACK): $(TRN)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available

# first train on the source dataset only and on target only:
results/prostate/cesource: OPT = --val_target_folders="$(SAUG_DATA_NORM)" --target_folders="$(SAUG_DATA_NORM)" --direct --batch_size 32  --target_losses="$(L_OR)" --target_dataset "/data/users/mathilde/ccnn/SAML/data/SB" \
	     --network UNet --model_weights="" --lr_decay 1 \

# then train tta with size only
results/prostate/enklsize: OPT =  --model_weights results/prostate/cesource/last.pkl --global_model  --batch_size 32 --l_rate 5e-4 --lr_decay 0.7  --target_losses="$(LSIZE)" \

# tta with other shape moments on target
results/prostate/enklsizecent: OPT =   --l_rate 5e-4 --update_mom_est --ind_mom 1 --target_losses="$(LSIZECentroid)" \

results/prostate/enklsizedist: OPT =  --l_rate 5e-4 --update_mom_est --ind_mom 1 --target_losses="$(LSIZEQuadDist)"  \


$(TRN) :
	$(CC) $(CFLAGS) main.py --valonly  --ontest --notent --regex_list "['Case22','Case17','Case26','Case05','Case02','Case07','Case08','Case12','Case15','Case20']" --batch_size 32 --n_class 2 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --wh 384 --metric_axis 1  --n_epoch 151 --dice_3d --l_rate 1e-4 --weight_decay 1e-4 --train_grp_regex="$(G_RGX)" --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(T_DATA)"\
                     --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(T_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@

