#!/bin/bash
#SBATCH -J StandardJob           
#SBATCH -c 2                     
#SBATCH --mem=8G                 
#SBATCH -p standard              
#SBATCH --gres=gpu:1             
#SBATCH --tmp=5G                 

conda init
# Run pytorch via python environment
conda activate  stars

#srun python  train_maskrcnn.py

#srun  python resume_training.py

##srun python main_megapose.py  --backbone 'maskrcnn'  --backbone_cfg 'configs/ycbv_rcnn.yaml'  --enc_layers 5 --dec_layers 5 --nheads 16 --resume 'updated_model_checkpoint.pth' --backbone_weights '../maskrcnn_utils/checkpoint_ff.pth'   --output_dir  'trained_models_f'
srun python  main.py 


#srun  python main_megapose.py --backbone 'maskrcnn'  --backbone_cfg 'configs/ycbv_rcnn.yaml'  --enc_layers 5 --dec_layers 5 --nheads 16 --resume 'trained_models_f/checkpoint.pth' --backbone_weights '../maskrcnn_utils/checkpoint_ff.pth' --inference_path 'atheel' --inference_output output   --inference


#srun python run_meapose_on_dataset.py
