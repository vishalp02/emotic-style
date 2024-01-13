import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename, splitext
from torchvision import transforms
from torchvision.utils import save_image
import net
import torch.distributed as dist
import torch.multiprocessing as mp

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

# Define a function to get a list of image files from a directory
def get_image_files_from_directory(directory):
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    return [os.path.join(directory, f) for f in image_files]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def is_below_size_threshold(file_path, size_threshold):
    # Check if the file size is below the specified threshold (in bytes)
    return os.path.getsize(file_path) < size_threshold

def is_grayscale_image(image):
    # Check if an image is grayscale
    return image.mode == 'L'

def run(rank, world_size):
    setup(rank, world_size)

    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument('--content_dir', type=str, help='Directory containing content images')
    parser.add_argument('--style_dir', type=str,  help='Directory containing style images')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--vgg', type=str, default='model/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='model/decoder_iter_160000.pth')
    parser.add_argument('--transform', type=str, default='model/transformer_iter_160000.pth')

    # Additional options
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output/s_mscoco',
                        help='Directory to save the output image(s)')

    # Advanced options
    args = parser.parse_args()
    
    # Maximum image size thresholds (in bytes)
    if 'mscoco' in args.content_dir: 
        # for mscoco images
        content_image_size_threshold = 500 * 1024  # 500KB
        style_image_size_threshold = 2 * 1024**2  # 2MB
    else:
        #for other images
        content_image_size_threshold = 2 * 1024**2  # 2MB
        style_image_size_threshold = 500 * 1024  # 500KB
 

    device = torch.device("cuda:{}".format(rank))

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # Load the model on a single GPU
    decoder = net.decoder
    transform = net.Transform(in_planes=512)
    vgg = net.vgg

    decoder.eval()
    transform.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    transform.load_state_dict(torch.load(args.transform))
    vgg.load_state_dict(torch.load(args.vgg))

    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

    norm.to(device)
    enc_1.to(device)
    enc_2.to(device)
    enc_3.to(device)
    enc_4.to(device)
    enc_5.to(device)
    transform.to(device)
    decoder.to(device)

    # content_tf = transforms.Compose([transforms.ToTensor()])
    # style_tf = transforms.Compose([transforms.ToTensor()])

    content_tf = test_transform()
    style_tf = test_transform()

    # Get a list of content and style image files from directories
    content_files = get_image_files_from_directory(args.content_dir)
    style_files = get_image_files_from_directory(args.style_dir)

    if len(content_files) == 0 or len(style_files) == 0:
        raise ValueError("No image files found in one or both of the specified directories.")

    # Initialize DistributedDataParallel
    decoder = nn.parallel.DistributedDataParallel(decoder, device_ids=[rank])
    transform = nn.parallel.DistributedDataParallel(transform, device_ids=[rank])
    norm = nn.parallel.DistributedDataParallel(norm, device_ids=[rank])
    enc_1 = nn.parallel.DistributedDataParallel(enc_1, device_ids=[rank])
    enc_2 = nn.parallel.DistributedDataParallel(enc_2, device_ids=[rank])
    enc_3 = nn.parallel.DistributedDataParallel(enc_3, device_ids=[rank])
    enc_4 = nn.parallel.DistributedDataParallel(enc_4, device_ids=[rank])
    enc_5 = nn.parallel.DistributedDataParallel(enc_5, device_ids=[rank])

    # Initialize skip counts
    skipped_content_images = 0
    skipped_style_images = 0
    skipped_grayscale_content_images = 0
    skipped_runtime_error_images = 0

    # Initialize image counter
    total_images_processed = 0

    with torch.no_grad():
        for i in range(rank, len(content_files), world_size):
            content_file = content_files[i]

            # Check content image size
            if not is_below_size_threshold(content_file, content_image_size_threshold):
                print(f'Skipping {content_file} due to size exceeding 2MB.')
                skipped_content_images += 1
                continue  # Skip images that are above the size threshold

            try:
                content = Image.open(content_file)

                # Check if the content image is grayscale (1 channel)
                if is_grayscale_image(content):
                    print(f'Skipping grayscale image {content_file}.')
                    skipped_grayscale_content_images += 1
                    continue

                content = content_tf(content)
                content = content.to(device).unsqueeze(0)

                style_file = style_files[i]

                # Check style image size
                if not is_below_size_threshold(style_file, style_image_size_threshold):
                    print(f'Skipping {style_file} due to size exceeding 500KB.')
                    skipped_style_images += 1
                    continue  # Skip images that are above the size threshold

                style = style_tf(Image.open(style_file))
                style = style.to(device).unsqueeze(0)

                for x in range(args.steps):
                    print('iteration ' + str(x))
                    Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
                    Content5_1 = enc_5(Content4_1)
                    Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
                    Style5_1 = enc_5(Style4_1)
                    content = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))
                    content.clamp(0, 255)

                content = content.cpu()

                output_name = '{:s}/{:s}{:s}'.format(
                    args.output, splitext(basename(content_file))[0], args.save_ext
                )
                save_image(content, output_name)
                del content
                del style
                del Content4_1
                del Content5_1
                del Style4_1
                del Style5_1
                torch.cuda.empty_cache()

                # Increment the total image counter
                total_images_processed += 1

            except RuntimeError as e:
                print(f'Error processing {content_file} or {style_file}: {e}')
                skipped_runtime_error_images += 1
                continue  # Skip images that cause a RuntimeError

    cleanup()

    # Print skip counts and total images processed
    print(f'Total images processed: {total_images_processed}')
    print(f'Skipped {skipped_content_images} content images above 2MB.')
    print(f'Skipped {skipped_style_images} style images above 500KB.')
    print(f'Skipped {skipped_grayscale_content_images} grayscale content images.')
    print(f'Skipped {skipped_runtime_error_images} images due to RuntimeError.')

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Use only one GPU - torch.cuda.device_count()
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
