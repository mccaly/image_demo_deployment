import argparse
import os.path as osp

import mmcv
import numpy as np
import cv2
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot


palette = np.random.randint(0, 255, (104, 3))
DEFAULT_PALETTE = palette




def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg Inference')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cpu', help='Device used for inference')
    args = parser.parse_args()
    return args
    
def compute_all_segments_area(image):
    """
    Computes the pixel area for each segment in the image.
    Args:
    - image: A 2D numpy array representing the segmented image (grayscale).
    
    Returns:
    - A dictionary with segment values as keys and their corresponding areas (in pixels) as values.
    """
    # Ensure image is grayscale (single channel)
    assert len(image.shape) == 2, "Image should be grayscale"

    unique_values, counts = np.unique(image, return_counts=True)
    return dict(zip(unique_values, counts))

def main():
    args = parse_args()
    
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    
    # Load class names from the provided file
    with open('./data/FoodSeg103/category_id.txt', 'r') as f:
        class_names = [line.strip().split('\t')[1] for line in f.readlines()]
        
    model.CLASSES = class_names
        

    # test a single image
    result = inference_segmentor(model, args.img)
    
    # show the results
    print("Inside image_demo DEFAULT_PALETTE.shape:", DEFAULT_PALETTE.shape)


    
    # Calculate segment areas
    segment_areas = compute_all_segments_area(result[0])

    # Map segment values to their respective class names
    segment_classes = {class_names[segment_value]: area for segment_value, area in segment_areas.items()}

    # Print segment areas with their class names
    for class_name, area in segment_classes.items():
        print(f"Segment for class '{class_name}': Area = {area} pixels")
        
    #show_result_pyplot(model, args.img, result, DEFAULT_PALETTE)
    class_names = model.CLASSES if model.CLASSES is not None else ['class_{}'.format(i) for i in range(104)]
    output_path = './output_images/segmented_result.jpg'
    show_result_pyplot(model, args.img, result, DEFAULT_PALETTE, out_file=output_path, class_names=class_names)
    
        # Assuming you've already loaded the model and have the segmentation result
    class_names = model.CLASSES if model.CLASSES is not None else ['class_{}'.format(i) for i in range(104)]



if __name__ == '__main__':
    main()

