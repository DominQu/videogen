from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def gen_seq(folder: Path, img_prefix: str):
    """Take images from folder and create image sequence from them"""
    images = list(folder.glob(f"{img_prefix}*"))
    img_count = len(images)
    loaded_img = []
    first_img = True
    sequence = np.zeros((1,1))
    for i, img in enumerate(images):
        data = np.array(Image.open(img))
        # plt.imshow(data)
        # plt.show()
        if first_img:
            height = data.shape[0]
            width = data.shape[1]
            ch = data.shape[2]
            sequence = np.zeros((height, img_count*width, ch), dtype=np.uint8)
            first_img = False
        sequence[:, i*width:(i+1)*width,:] = data
    img_seq = Image.fromarray(sequence, mode="RGB")
    img_seq.save("sequence.png")
    # plt.imshow(sequence)
    # plt.show()

def gen_img(folder, origin, pred):
    original = list(folder.glob(f"{origin}*"))
    predicted = list(folder.glob(f"{pred}*"))
    img_array = 255 * np.ones((3*64+2, 10*64, 3), dtype=np.uint8)
    images = [original[:10], original[10:], predicted[10:]]
    for j in range(3):
        for i in range(10):
            img_array[j*64+j:(j+1)*64+j,i*64:(i+1)*64] = np.array(Image.open(images[j][i]))
        img = Image.fromarray(img_array[j*64+j:(j+1)*64+j,:])
        img.save(f"seq_{pred}_{j}.png")
    full_img = Image.fromarray(img_array)
    full_img.save(f"seq_{pred}.png")

def gen_penn_img(folder, origin, pred):
    original = list(folder.glob(f"{origin}*"))
    predicted = list(folder.glob(f"{pred}*"))
    img_array = 255 * np.ones((3*64+2, 10*64, 3), dtype=np.uint8)
    images = [original[:10], original[10:], predicted[10:]]
    for j in range(3):
        for i in range(10):
            img_array[j*64+j:(j+1)*64+j,i*64:(i+1)*64] = np.array(Image.open(images[j][i]))
        array = 255 * np.ones((2*64+1, 5*64, 3), dtype=np.uint8)
        array[:64] = img_array[j*64+j:(j+1)*64+j, :320]
        array[65:] = img_array[j*64+j:(j+1)*64+j, 320:]
        img = Image.fromarray(array)
        img.save(f"seq_{pred}_{j}_5.png")
        # img = Image.fromarray(img_array[j*64+j:(j+1)*64+j,:320])

        # img.save(f"seq_{pred}_{j}_1.png")
        # img = Image.fromarray(img_array[j*64+j:(j+1)*64+j,320:])
        # img.save(f"seq_{pred}_{j}_2.png")

    full_img = Image.fromarray(img_array)
    full_img.save(f"seq_{pred}.png")

if __name__ == "__main__":
    dirp = Path("models/run_19_11_2022_19_02_02/outputs/img")
    img_prefix = "orig_epoch_2"
    num = "seq"
    gen_penn_img(dirp, f"orig_epoch_{num}", f"pred_epoch_{num}")