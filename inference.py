import cv2
import numpy as np 
import os 
from sklearn.metrics import average_precision_score, precision_recall_curve

import torch 
from torchvision import transforms

from emotic import Emotic 


def process_images(context_norm, body_norm, image_context_path=None, image_context=None, image_body=None, bbox=None):
  ''' Prepare context and body image. 
  :param context_norm: List containing mean and std values for context images. 
  :param body_norm: List containing mean and std values for body images. 
  :param image_context_path: Path of the context image. 
  :param image_context: Numpy array of the context image.
  :param image_body: Numpy array of the body image. 
  :param bbox: List to specify the bounding box to generate the body image. bbox = [x1, y1, x2, y2].
  :return: Transformed image_context tensor and image_body tensor.
  '''
  if image_context is None and image_context_path is None:
    raise ValueError('both image_context and image_context_path cannot be none. Please specify one of the two.')
  if image_body is None and bbox is None: 
    raise ValueError('both body image and bounding box cannot be none. Please specify one of the two')

  if image_context_path is not None:
    image_context =  cv2.cvtColor(cv2.imread(image_context_path), cv2.COLOR_BGR2RGB)
  
  if bbox is not None:
    image_body = image_context[bbox[1]:bbox[3],bbox[0]:bbox[2]].copy()
  
  image_context = cv2.resize(image_context, (224,224))
  image_body = cv2.resize(image_body, (224,224))
  
  test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
  context_norm = transforms.Normalize(context_norm[0], context_norm[1])  
  body_norm = transforms.Normalize(body_norm[0], body_norm[1])

  image_context = context_norm(test_transform(image_context)).unsqueeze(0)
  image_body = body_norm(test_transform(image_body)).unsqueeze(0)

  return image_context, image_body  


def test_scikit_ap(cat_preds, cat_labels, ind2cat):
  ''' Calculate average precision per emotion category using sklearn library.
  :param cat_preds: Categorical emotion predictions. 
  :param cat_labels: Categorical emotion labels. 
  :param ind2cat: Dictionary converting integer index to categorical emotion.
  :return: Numpy array containing average precision per emotion category.
  '''
  ap = np.zeros(26, dtype=np.float32)
  for i in range(26):
    ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
    print ('Category %16s %.5f' %(ind2cat[i], ap[i]))
  print ('Mean AP %.5f' %(ap.mean()))
  return ap 

def test_vad(cont_preds, cont_labels, ind2vad):
  ''' Calcaulate VAD (valence, arousal, dominance) errors. 
  :param cont_preds: Continuous emotion predictions. 
  :param cont_labels: Continuous emotion labels. 
  :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
  :return: Numpy array containing mean absolute error per continuous emotion dimension. 
  '''
  vad = np.zeros(3, dtype=np.float32)
  for i in range(3):
    vad[i] = np.mean(np.abs(cont_preds[i, :] - cont_labels[i, :]))
    print ('Continuous %10s %.5f' %(ind2vad[i], vad[i]))
  print ('Mean VAD Error %.5f' %(vad.mean()))
  return vad


def infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context_path=None, image_context=None, image_body=None, bbox=None, to_print=True):
  ''' Perform inference over an image. 
  :param context_norm: List containing mean and std values for context images. 
  :param body_norm: List containing mean and std values for body images. 
  :param ind2cat: Dictionary converting integer index to categorical emotion. 
  :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
  :param device: Torch device. Used to send tensors to GPU if available.
  :param image_context_path: Path of the context image. 
  :param image_context: Numpy array of the context image.
  :param image_body: Numpy array of the body image. 
  :param bbox: List to specify the bounding box to generate the body image. bbox = [x1, y1, x2, y2].
  :param to_print: Variable to display inference results.
  :return: Categorical Emotions list and continuous emotion dimensions numpy array.
  '''
  image_context, image_body = process_images(context_norm, body_norm, image_context_path=image_context_path, image_context=image_context, image_body=image_body, bbox=bbox)

  model_context, model_body, emotic_model = models

  with torch.no_grad():
    image_context = image_context.to(device)
    image_body = image_body.to(device)
    
    pred_context = model_context(image_context)
    pred_body = model_body(image_body)
    pred_cat_a, pred_cont_a = emotic_model(pred_context, pred_body)
    pred_cat = pred_cat_a.squeeze(0)
    pred_cont = pred_cont_a.squeeze(0).to("cpu").data.numpy()

    bool_cat_pred = torch.gt(pred_cat, thresholds)
  
  cat_emotions = list()
  for i in range(len(bool_cat_pred)):
    if bool_cat_pred[i] == True:
      cat_emotions.append(ind2cat[i])

  if to_print == True:
    print ('\n Image predictions')
    print ('Continuous Dimnesions Predictions') 
    for i in range(len(pred_cont)):
      print ('Continuous %10s %.5f' %(ind2vad[i], 10*pred_cont[i]))
    print ('Categorical Emotion Predictions')
    for emotion in cat_emotions:
      print ('Categorical %16s' %(emotion))
  
  return cat_emotions, 10*pred_cont, pred_cat_a, pred_cont_a


def inference_emotic(images_list, model_path, result_path, context_norm, body_norm, ind2cat, ind2vad, args):
  ''' Infer on list of images defined in a text file. Save the results in inference_file.txt in the directory specified by the result_path. 
  :param images_list: Text file specifying the images and their bounding box values to conduct inference. A row in the file is Path_of_image x1 y1 x2 y2. 
  :param model_path: Directory path to load models and val_thresholds to perform inference.
  :param result_path: Directory path to save the results (text file containig categorical emotion and continuous emotion dimension prediction per image).
  :param context_norm: List containing mean and std values for context images. 
  :param body_norm: List containing mean and std values for body images. 
  :param ind2cat: Dictionary converting integer index to categorical emotion. 
  :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
  :param args: Runtime arguments.
  '''


  with open(images_list, 'r') as f:
    lines = f.readlines()
  
  ################
  num_images = len(lines)
  print(num_images)
  ################

  device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
  thresholds = torch.FloatTensor(np.load(os.path.join(result_path, 'val_thresholds.npy'))).to(device) 
  model_context = torch.load(os.path.join(model_path,'model_context1.pth')).to(device)
  model_body = torch.load(os.path.join(model_path,'model_body1.pth')).to(device)
  emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth')).to(device)
  model_context.eval()
  model_body.eval()
  emotic_model.eval()
  models = [model_context, model_body, emotic_model]

  result_file = os.path.join(result_path, 'inference_list.txt')
  with open(result_file, 'w') as f:
    pass
  

  #################
  cat_preds = np.zeros((num_images, 26))
  cont_preds = np.zeros((num_images, 3))

  cat_labels = []
  cont_labels = []
  #################

  for idx, line in enumerate(lines):
    image_context_path, x1, y1, x2, y2, emotion, valence, arousal, dominance  = line.split('\n')[0].split(' ')
    bbox = [int(x1), int(y1), int(x2), int(y2)]
    pred_cat, pred_cont, pred_cat_s, pred_cont_s = infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context_path=image_context_path, bbox=bbox)
    

    ##################
    emotion_list = [0] * len(ind2cat)
    if emotion in ind2cat.values():
        emotion_index = list(ind2cat.values()).index(emotion)
        emotion_list[emotion_index] = 1

    cat_labels.append(emotion_list)
    cat_preds[ idx : (idx + pred_cat_s.shape[0]), :] = pred_cat_s.to("cpu").data.numpy()

    # Create a tuple of (valence, arousal, dominance) and add it to the list
    vad_tuple = (float(valence), float(arousal), float(dominance))
    cont_labels.append(vad_tuple)

    cont_preds[ idx : (idx + pred_cont_s.shape[0]), :] = pred_cont_s.to("cpu").data.numpy() * 10

    idx = idx + pred_cat_s.shape[0]
    ##################


    write_line = list()
    write_line.append(image_context_path)
    for emotion in pred_cat:
      write_line.append(emotion)
    for continuous in pred_cont:
      write_line.append(str('%.4f' %(continuous)))
    write_line = ' '.join(write_line) 
    with open(result_file, 'a') as f:
      f.writelines(write_line)
      f.writelines('\n')
  
  ##################
  cat_labels = np.array(cat_labels)
  cont_labels = np.array(cont_labels)

  cat_preds = cat_preds.transpose()
  cat_labels = cat_labels.transpose()
  cont_preds = cont_preds.transpose()
  cont_labels = cont_labels.transpose()

  print("____________________for all____________________")
  test_scikit_ap(cat_preds, cat_labels, ind2cat)
  test_vad(cont_preds, cont_labels, ind2vad)
#####################
