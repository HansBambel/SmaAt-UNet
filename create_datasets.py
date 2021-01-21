import h5py
import numpy as np
from tqdm import tqdm


def create_dataset(input_length, image_ahead, rain_amount_thresh):
    with h5py.File("data/precipitation/RAD_NL25_RAC_5min_train_test_2016-2019.h5", "r", rdcc_nbytes=1024 ** 3) as orig_f:
        train_images = orig_f["train"]["images"]
        train_timestamps = orig_f["train"]["timestamps"]
        test_images = orig_f["test"]["images"]
        test_timestamps = orig_f["test"]["timestamps"]
        print("Train shape", train_images.shape)
        print("Test shape", test_images.shape)
        imgSize = train_images.shape[1]
        num_pixels = imgSize * imgSize

        filename = f"data/precipitation/train_test_2016-2019_input-length_{input_length}_img-ahead_{image_ahead}_rain-threshhold_{int(rain_amount_thresh * 100)}.h5"
        with h5py.File(filename, "w", rdcc_nbytes=1024 ** 3) as f:
            train_set = f.create_group("train")
            test_set = f.create_group("test")
            train_image_dataset = train_set.create_dataset("images",
                                                           shape=(1, input_length + image_ahead, imgSize, imgSize),
                                                           maxshape=(None, input_length + image_ahead, imgSize, imgSize),
                                                           dtype='float32', compression="gzip", compression_opts=9)
            train_timestamp_dataset = train_set.create_dataset("timestamps", shape=(1, input_length + image_ahead, 1),
                                                               maxshape=(None, input_length + image_ahead, 1),
                                                               dtype=h5py.special_dtype(vlen=str), compression="gzip",
                                                               compression_opts=9)
            test_image_dataset = test_set.create_dataset("images", shape=(1, input_length + image_ahead, imgSize, imgSize),
                                                         maxshape=(None, input_length + image_ahead, imgSize, imgSize),
                                                         dtype='float32', compression="gzip", compression_opts=9)
            test_timestamp_dataset = test_set.create_dataset("timestamps", shape=(1, input_length + image_ahead, 1),
                                                             maxshape=(None, input_length + image_ahead, 1),
                                                             dtype=h5py.special_dtype(vlen=str), compression="gzip",
                                                             compression_opts=9)

            origin = [[train_images, train_timestamps], [test_images, test_timestamps]]
            datasets = [[train_image_dataset, train_timestamp_dataset], [test_image_dataset, test_timestamp_dataset]]
            for origin_id, (images, timestamps) in enumerate(origin):
                image_dataset, timestamp_dataset = datasets[origin_id]
                first = True
                for i in tqdm(range(input_length + image_ahead, len(images))):
                    # If threshold of rain is bigger in the target image: add sequence to dataset
                    if np.sum(images[i] > 0) >= num_pixels * rain_amount_thresh:
                        imgs = images[i - (input_length + image_ahead):i]
                        timestamps_img = timestamps[i - (input_length + image_ahead):i]
                        #                     print(imgs.shape)
                        #                     print(timestamps_img.shape)
                        # extend the dataset by 1 and add the entry
                        if first:
                            first = False
                        else:
                            image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                            timestamp_dataset.resize(timestamp_dataset.shape[0] + 1, axis=0)
                        image_dataset[-1] = imgs
                        timestamp_dataset[-1] = timestamps_img


if __name__ == "__main__":
    create_dataset(input_length=12, image_ahead=6, rain_amount_thresh=0.2)
    create_dataset(input_length=12, image_ahead=6, rain_amount_thresh=0.5)