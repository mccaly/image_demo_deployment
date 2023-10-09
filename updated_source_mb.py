print('Script started...')

import argparse
import os.path as osp
import sys
import requests
from tempfile import NamedTemporaryFile

import mmcv
import numpy as np
import cv2
import modelbit
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot


palette = np.random.randint(0, 255, (104, 3))
DEFAULT_PALETTE = palette

def fetch_image_from_url(img_url):
    response = requests.get(img_url, stream=True)
    response.raise_for_status()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
        return f.name


def parse_args(img_url):
    parser = argparse.ArgumentParser(description='MMSeg Inference')
    
    #parser.add_argument('config', help='Config file')
    #parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img_url', help='Image URL for processing')
    parser.add_argument('--device', default='cpu', help='Device used for inference')
    parser.add_argument("--modelbit-only", action="store_true", help="Run only modelbit commands")
    
    args = parser.parse_args()
    img_path = fetch_image_from_url(img_url)
    return argparse.Namespace(img=img_path, config='./checkpoints/swin_small/upernet_swin_small_patch4_window7_512x1024_80k.py', checkpoint='./checkpoints/swin_small/iter_80000.pth', device='cpu')

    
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

def run_segment(img_url=None):
    #make args hard coded and pass the URL as a parameter in main
    #args = parse_args(img_url)

     # If img_url isn't provided as an argument, fetch it from command-line arguments
    if not img_url:
        args = parse_args()
        img_url = args.img_url
    else:
        img_path = fetch_image_from_url(img_url)
        args = argparse.Namespace(img=img_path, config='./checkpoints/swin_small/upernet_swin_small_patch4_window7_512x1024_80k.py', checkpoint='./checkpoints/swin_small/iter_80000.pth', device='cpu')
    

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    
    # Load class names from the provided file
    with open('./category_id.txt', 'r') as f:
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
    if "--modelbit-only" in sys.argv:
        # Your modelbit commands here
        print("Running modelbit commands...")
        print('Executing modelbit commands...')
        mb = modelbit.login()
        mb.add_common_files(".")
        mb.deploy(run_segment, python_version="3.7.16", python_packages=[
        "addict==2.4.0",
        "appdirs==1.4.4",
        "attrs==23.1.0",
        "boto3==1.28.45",
        "botocore==1.31.45",
        "build==0.10.0",
        "certifi==2022.12.7",
        "charset-normalizer==3.2.0",
        "cityscapesScripts==2.2.2",
        "click==8.1.6",
        "codecov==2.1.13",
        "colorama==0.4.6",
        "coloredlogs==15.0.1",
        "coverage==7.2.7",
        "cycler==0.11.0",
        "Cython==3.0.0",
        "exceptiongroup==1.1.2",
        "filelock==3.12.2",
        "flake8==5.0.4",
        "fonttools==4.38.0",
        "fsspec==2023.1.0",
        "huggingface-hub==0.16.4",
        "humanfriendly==10.0",
        "idna==3.4",
        "importlib-metadata==4.2.0",
        "iniconfig==2.0.0",
        "interrogate==1.5.0",
        "isort==4.3.21",
        "Jinja2==3.1.2",
        "jmespath==1.0.1",
        "kiwisolver==1.4.4",
        "markdown-it-py==2.2.0",
        "MarkupSafe==2.1.3",
        "matplotlib==3.5.3",
        "mccabe==0.7.0",
        "mdurl==0.1.2",
        "mmcv-full==1.3.0",
        "mmengine==0.8.4",
        "mmsegmentation==1.1.1",
        "modelbit==0.28.7",
        "numpy==1.21.6",
        "opencv-python==4.8.0.76",
        "packaging==23.1",
        "pandas==1.3.5",
        "Pillow==9.5.0",
        "pkginfo==1.9.6",
        "platformdirs==3.10.0",
        "pluggy==1.2.0",
        "prettytable==3.7.0",
        "py==1.11.0",
        "pycodestyle==2.9.1",
        "pycryptodomex==3.18.0",
        "pyflakes==2.5.0",
        "Pygments==2.16.1",
        "pyparsing==3.1.1",
        "pyproject_hooks==1.0.0",
        "pyquaternion==0.9.9",
        "pytest==7.4.0",
        "python-dateutil==2.8.2",
        "pytz==2023.3",
        "PyYAML==6.0.1",
        "regex==2023.8.8",
        "requests==2.31.0",
        "rich==13.5.2",
        "s3transfer==0.6.2",
        "safetensors==0.3.2",
        "scipy==1.7.3",
        "six==1.16.0",
        "tabulate==0.9.0",
        "termcolor==2.3.0",
        "terminaltables==3.1.10",
        "texttable==1.6.7",
        "timm==0.9.5",
        "tokenizers==0.13.3",
        "toml==0.10.2",
        "tomli==2.0.1",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "tqdm==4.66.1",
        "transformers==4.30.2",
        "types-pkg-resources==0.1.3",
        "types-PyYAML==6.0.12.11",
        "types-requests==2.31.0.2",
        "types-urllib3==1.26.25.14",
        "typing==3.7.4.3",
        "typing_extensions==4.7.1",
        "urllib3==1.26.16",
        "wcwidth==0.2.6",
        "xdoctest==1.1.1",
        "yapf==0.33.0",
        "zipp==3.15.0",
        "zstandard==0.21.0"
            ])
        print('Script ended.')
        sys.exit(0)  # Exit after executing the modelbit commands

    elif len(sys.argv) < 2:
        print("Please provide the image URL as an argument.")
        sys.exit(1)
    else:
        img_url = sys.argv[1]
        run_segment(img_url)