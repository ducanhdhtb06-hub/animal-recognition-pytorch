from argparse import ArgumentParser
import torch
from simplenetwork import simpleCNN
import cv2
import torch.nn as nn
import numpy as np
from torchsummary import summary
def get_args():
    parser = ArgumentParser("CNN inference")
    parser.add_argument("--image_size", type=int, default=224, help="size of each image dimension")
    parser.add_argument("--image_path", type= str, default=None)
    parser.add_argument("--checkpoint", type=str, default="train_models/best_model.pt", help="train model")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    args = get_args()
    model = simpleCNN()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else :
        device = torch.device("cpu")
    model = simpleCNN(10).to(device)
    if args.checkpoint :
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    else :
        print("No checkpoint found")
        exit()
    model.eval()
    summary(model, (3, 224, 224))
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image , cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))/255.0
    image = image[None, : ,:, :]
    image = torch.from_numpy(image).to(device).float()
    sortmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        print(output)
        probs = sortmax(output)
        print(probs)

    max_idx = torch.argmax(probs)
    print(categories[max_idx])
    cv2.imshow("{}:{:.2f}%".format(categories[max_idx], probs[0,max_idx]), ori_image )
    cv2.waitKey(0)


