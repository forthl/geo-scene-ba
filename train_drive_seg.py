import lightning.pytorch as pl
import torch
import torch.utils.data as data
#from open3d._ml3d.datasets import NuScenes

from nuscenes.nuscenes import NuScenes

from src.data.nuscenes_data_module import nuScenes



from src.modules.test_pytorch import LitAutoEncoder, Encoder, Decoder

def main():
    
    checkpoint = ''
    
    # traverse root directory, and list directories as dirs and files as files
    # for root, dirs, files in os.walk("./checkpoints/lightning_logs/"):
    #     path = root.split(os.sep)
    #     for i, file in enumerate(files):
    #         if file.endswith('.ckpt'):
    #             checkpoint = root + '/' + file
    #             break
    
    nusc =  NuScenes(version='v1.0-mini', dataroot='./data/nuScenes/', verbose=True)
    train_data = nuScenes(nusc)

    torch.set_float32_matmul_precision('medium')

    data_loader_train = data.DataLoader(train_data, batch_size=16, collate_fn=nuScenes.collate_fn, shuffle=True, num_workers=1)

    # model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # train model
    trainer = pl.Trainer(accelerator="gpu", devices=1, default_root_dir="./checkpoints/", log_every_n_steps=1)
    
    if len(checkpoint) > 0:
        trainer.fit(model=autoencoder, train_dataloaders=data_loader_train, ckpt_path=checkpoint)
    else:    
        trainer.fit(model=autoencoder, train_dataloaders=data_loader_train)
    
    # trainer.test(model=autoencoder, dataloaders=data_loader_test)
    

if __name__ == "__main__":
    main()
