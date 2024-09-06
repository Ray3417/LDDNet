import os
import cv2
from dataset.dataloader import GenerateDataset
from tqdm import tqdm
from config import param as option
from model.model import get_model
from utils import set_seed

if __name__ == '__main__':
    set_seed(option)
    generate_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]
    model = get_model(option)
    model.eval()

    result_save_path = option['eval_save_path'] + 'soft_gt/'
    os.makedirs(result_save_path, exist_ok=True)
    generate_root = option['paths']['image_root']
    generate_loader = GenerateDataset(generate_root, option['trainsize'])

    for i in tqdm(range(generate_loader.size)):
        image, HH, WW, name = generate_loader.load_data()
        name = name.replace('jpg','png')
        image = image.cuda()
        res, _ = model.forward(image, (HH,WW))
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(result_save_path + name, res)

    print('Successful, generate in {}'.format(result_save_path))



