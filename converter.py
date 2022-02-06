import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import subprocess
import shutil
import math
import cv2
import pyscripts.model as model
import PIL
import glob
import onnx
import onnxoptimizer
import torch
import numpy as np
import onnxruntime as ort
import tensorflow.compat.v1 as tf
import pyscripts.infer as infer
from PIL import Image
from pyscripts.realesrgan import RealESRGAN
from pyscripts.deblur_namespace import DeblurNamespace
from cv2 import dnn_superres
tf.disable_v2_behavior()
scales = [2, 4, 8]
onnx_model = onnx.load("./ai/deepbump/deepbump256.onnx")
passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
optimized_model = onnxoptimizer.optimize(onnx_model, passes)
onnx.save(optimized_model, "./ai/deepbump/deepbump256.onnx")

def ai_dconvolution(iterations):
    if iterations>10:
        print("Belive me, you don't need so much iterations. 10 is maximum.")
        print('Variable "Iterations" was set to 10')
        iterations = 10
    count = cv2.cuda.getCudaEnabledDeviceCount()
    if count==0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    try:
        args = DeblurNamespace(model='color',
                               phase='test',
                               datalist='./datalist/datalist_gopro.txt',
                               batch_size=16,
                               epoch=4000,
                               learning_rate=1e-4,
                               gpu=-1,
                               height=2048,
                               width=2048,
                               input_path='./',
                               output_path='./')

        deblur = model.DEBLUR(args)
    except:
        print("Something went wrong during preparing of model")
        return
    for i in range(0, iterations):
        try:
            print("Deconvolution iteration: "+str(i+1))
            deblur.test(args.height, args.width, args.input_path, args.output_path)
        except:
            print("Something went wrong during model execution")
            return

def upscale_image(path_to_image, scaling_factor):
    if "watchable" in path_to_image:
        return
    if scaling_factor not in scales:
        print("Sorry, it's impossible to use your scaling factor")
        print("Allowed are: 2, 4, 8")
        if scaling_factor>6:
            scaling_factor = 8
        elif scaling_factor>4 and scaling_factor<6:
            scaling_factor = 4
        else:
            scaling_factor = 2
        print("We've decided to use scaling factor: "+str(scaling_factor))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=scaling_factor)
    model.load_weights('./ai/upscaling_weights/RealESRGAN-x'+str(scaling_factor)+'.pth')
    image = Image.open(path_to_image).convert('RGB')
    print("Processing Image: " + path_to_image)
    sr_image = model.predict(image)
    os.remove(path_to_image)
    sr_image.save(path_to_image)
    print("Upscaling of " + path_to_image + " done!")

def make_normalmap(img_src, bump_path, overlap):
    #based on https://github.com/HugoTini/DeepBump
    img = np.array(Image.open(img_src)) / 255.0
    img = np.transpose(img, [2, 0, 1])
    img = np.mean(img[0:3], axis=0, keepdims=True)
    print('Tilling of texture: ', img_src)
    tile_size = 256
    overlaps = {'small': tile_size // 6, 'medium': tile_size // 4, 'large': tile_size // 2}
    stride_size = tile_size - overlaps[overlap]
    tiles, paddings = infer.tiles_split(img, (tile_size, tile_size),(stride_size, stride_size))
    print('Generating normal map for texture: ', img_src)
    ort_session = ort.InferenceSession("./ai/deepbump/deepbump256.onnx")
    pred_tiles = infer.tiles_infer(tiles, ort_session)
    print('Merging tiles of texture: ', img_src)
    pred_img = infer.tiles_merge(pred_tiles, (stride_size, stride_size), (3, img.shape[1], img.shape[2]), paddings)
    pred_img = pred_img.transpose((1, 2, 0))
    pred_img = Image.fromarray((pred_img * 255.0).astype(np.uint8))
    pred_img.save(bump_path[:len(bump_path)-4]+"_watchable.png")
    pred_img.save(bump_path, compression=None)

def next_pow_of_two(x):
    a=math.ceil(math.log(x, 2))
    return int(math.pow(2.0, a))

def is_imagefile(i):
    i = i.lower()
    if i.endswith('.bmp') or i.endswith('.tga') or i.endswith('.png') or i.endswith(".jpeg") or i.endswith(".jpg"):
        return True
    else:
        return False

def reset_picture(x):
    #converts the length and width values of the image to the nearest power of two
    try:
        img1 = Image.open(x)
    except:
        print("Something went wrong during opening: "+x)
        return
    width, height = img1.size
    img1 = img1.resize((next_pow_of_two(width),next_pow_of_two(height)))
    img1.save(x)

def convert(string_modelpath, studiomdl_fullpath, compiled_models, upscaling=True, bump_maps=True, scale_factor=4, deconvolution=True, deconvolution_iterations=4):
    sourcepath = os.getcwd()
    if string_modelpath.endswith(".mdl"):
        #Creating File System
        head, tail = os.path.split(string_modelpath)
        listf = os.listdir(sourcepath)
        if str(tail) not in listf:
           shutil.copyfile(string_modelpath, sourcepath + '\\' +str(tail))
        vtf_names = []
        truename = str(tail)[:len(tail) - 4]
        head += '\\'
        if not os.path.exists(sourcepath+'\\'+"ConvertedModels"+'\\'+truename):
            args = sourcepath + '//' + 'mdldec.exe ' + truename+".mdl"
            print('Model decompilation starts')
            subprocess.call(args, shell=False)
            print('Decompilation Done')
            if os.path.exists(sourcepath + '\\'+"ConvertedModels"):
                os.chdir("ConvertedModels")
            else:
                os.mkdir("ConvertedModels")
                os.chdir("ConvertedModels")
            if not os.path.exists(sourcepath + "ConvertedModels" + '\\' + truename):
                os.mkdir(truename)
            os.chdir(truename)
            if not os.path.exists(sourcepath + "ConvertedModels" + '\\' + truename + '\\' + "materials"):
                os.mkdir("materials")
            if not os.path.exists(sourcepath + "ConvertedModels" + '\\' + truename + '\\' + "models"):
                os.mkdir("models")
            os.chdir("materials")
            if not os.path.exists(
                    sourcepath + "ConvertedModels" + '\\' + truename + '\\' + "materials" + '\\' + "models"):
                os.mkdir("models")
            os.chdir("models")
            if not os.path.exists(
                    sourcepath + "ConvertedModels" + '\\' + truename + '\\' + "materials" + '\\' + "models" + '\\' + "conv_graphics"):
                os.mkdir("conv_graphics")
            os.chdir(sourcepath)
            if not os.path.exists(sourcepath +'\\'+ "ModelFiles"+'\\'): os.mkdir("ModelFiles")
            os.chdir("ModelFiles")
            if not os.path.exists(sourcepath + "ModelFiles" + '\\' + truename):
                os.mkdir(truename)
            os.chdir(sourcepath)

            # resetting sizes of images to powes of 2
            for i in os.listdir(sourcepath):
                if is_imagefile(i):
                    if upscaling:
                        upscale_image(i, scaling_factor=scale_factor)
                        reset_picture(i)
            if deconvolution:
                ai_dconvolution(iterations=deconvolution_iterations)
            if bump_maps:
                for i in os.listdir(sourcepath):
                    if is_imagefile(i):
                        make_normalmap(i, i[:len(i)-4]+'_normal'+'.tga', 'large')

            # generating of VTF files
            print('Generating VTF files')
            for i in os.listdir(sourcepath):
                if is_imagefile(i) and not "_normal" in i:
                    vtf_names.append(i[:len(i) - 4])
                    args_nova = "VTFCmd.exe -file " + i+' -rsharpen "CONTRASTMORE" -format "ABGR8888" '
                    subprocess.call(args_nova)
                elif is_imagefile(i) and "_normal" in i:
                    vtf_names.append(i[:len(i) - 4])
                    args_nova = 'VTFCmd.exe -format "A8" -rsharpen "CONTRASTMORE"  -normal -file ' + i
                    subprocess.call(args_nova)
            for i in os.listdir(sourcepath):
                if is_imagefile(i):
                    shutil.move(sourcepath + '\\' + i, sourcepath + '\\' + 'ModelFiles' + '\\' + truename + '\\')
            # generating of  VMT files
            for i in vtf_names:
                if not "_normal" in i:
                    filename = i + '.vmt'
                    f = open(filename, "w+")
                    if bump_maps != True:
                        f.write('"VertexlitGeneric"')
                        f.write('{')
                        f.write('"$basetexture" "models/conv_graphics/' + i + '"')
                        f.write('}')
                    else:
                        f.write("VertexLitGeneric" + '\n')
                        f.write('{' + '\n')
                        f.write('"$basetexture" "models/conv_graphics/' + i + '"' + '\n')
                        f.write('"$surfaceprop" "zombieflesh"'+ '\n')
                        f.write('"$bumpmap" "models/conv_graphics/' + i +"_normal"+ '"' + '\n')
                        f.write('"$ssbump" 1' + '\n')
                        f.write('"$SSBumpMathFix" 1' + '\n')
                        f.write('}')
                    f.close()

            for i in vtf_names:
                shutil.move(sourcepath + '\\' + i + '.vtf',
                            sourcepath + '\\' + 'ConvertedModels\\' + truename + '\\materials\\models\\conv_graphics')
                if not "_normal" in i:
                    shutil.move(sourcepath + '\\' + i + '.vmt', sourcepath + '\\' + 'ConvertedModels\\' + truename + '\\materials\\models\\conv_graphics')
            #Redacting .QC file
            filename_nova = truename + '.qc'
            f = open(filename_nova, "r+")
            d = f.readlines()
            f.seek(0)
            for i in d:
                if not 'cdtexture' in i:
                    f.write(i)
            f.truncate()
            f.write('$cdmaterials "models\conv_graphics"')
            f.close()

            args_nova = studiomdl_fullpath + ' ' + truename + '.qc'
            print(args_nova)
            subprocess.call(args_nova)
            #Getting data from "models" folder
            list_modelsfiles = os.listdir(compiled_models)
            for i in list_modelsfiles:
                if truename in i or truename.upper() in i or truename.lower() in i:
                    shutil.move(compiled_models + '\\' + i, sourcepath + '\\' + 'ConvertedModels\\' + truename + '\\' + "models\\")
            list_of_files = os.listdir(sourcepath)
            for i in list_of_files:
                if i.endswith('.smd') or i.endswith('.qc'):
                    shutil.move(sourcepath + '\\' + i, sourcepath + '\\' + "ModelFiles" + '\\' + truename + '\\')


def argsparser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='-1', type=str, required=True)
    parser.add_argument('-studiomdl', '--studiomdl', default='-1', required=True, type=str)
    parser.add_argument('-gamemodels', '--compiled', default='-1', required=True, type=str)
    parser.add_argument('-upscaling', '--upscaling', default=True, required=True, type=bool)
    parser.add_argument('-nmap', '--normalmaps', default=True, required=True, type=bool)
    parser.add_argument('-sf', '--scaling_factor', default=4, type=int)
    parser.add_argument('-dc', '--deconvolution', default=True, type=bool, required=True)
    parser.add_argument('-it', '--iterations', default=4, type=int)
    return parser


def main():
    parser = argsparser()
    namespace = parser.parse_args(sys.argv[1:])
    input = format(namespace.input)
    studiomdl = format(namespace.studiomdl)
    compiled_models = format(namespace.compiled)
    upscaling = format(namespace.upscaling)
    nmaps = format(namespace.normalmaps)
    sf = format(namespace.scaling_factor)
    deconvolution = format(namespace.deconvolution)
    iters = int(format(namespace.iterations))
    if input == '-1' or studiomdl == '-1' or compiled_models == '-1':
        print("Wrong input arguments")
        return
    else:
        studiomdl = os.path.normpath(studiomdl)
        compiled_models = os.path.normpath(compiled_models)
        input = os.path.normpath(input)
        if os.path.exists(studiomdl) and os.path.exists(compiled_models) and os.path.exists(input):
            if input.endswith(".mdl"):
                convert(string_modelpath=input,
                        studiomdl_fullpath=studiomdl,
                        compiled_models=compiled_models,
                        upscaling=eval(upscaling),
                        bump_maps=eval(nmaps),
                        scale_factor=int(sf),
                        deconvolution=eval(deconvolution),
                        deconvolution_iterations=iters)
            elif os.path.isdir(input):
                files = os.listdir(input)
                for i in files:
                    if i.endswith(".mdl"):
                        print(input + "\\" + i)
                        shutil.move(input + '\\' + i, os.getcwd())
                        convert(string_modelpath=os.getcwd() + "//" + i,
                                studiomdl_fullpath=studiomdl,
                                compiled_models=compiled_models,
                                upscaling=eval(upscaling),
                                bump_maps=eval(nmaps),
                                scale_factor=int(sf),
                                deconvolution=eval(deconvolution),
                                deconvolution_iterations=iters)
                        shutil.move(os.getcwd() + '\\' + i, input + '\\' + i)

    print("Done!")

if __name__ == '__main__':
    main()

#example run: python converter.py --input cactus.mdl  --studiomdl   "D:\\SteamLibrary\\steamapps\\common\\Team Fortress 2\\bin\\studiomdl.exe" --compiled "D:\\SteamLibrary\\steamapps\\common\\Team Fortress 2\\tf\\models\\"  --upscaling True --scaling_factor 4 --normalmaps True --deconvolution True --iterations 1