#TODO
#idet multi procesinga kad dar greiciau suktusi
#isbandyti ant kameru
#pasidometi ar nera geresniu budu capturint gal vertetu padaryt atskir programele kuri dumpintu screenus ir atskira kuri juos apdorotu
import numpy as np
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import time
import cv2
import mss
#envirmento setingai kad nemestu erroru del silpno cpu ir kad mss neverktu kad neturim teises imt ramu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['QT_X11_NO_MITSHM'] = '1'
#basic kintamieji
start_time = time.time()
title = "Screen-grab-test"
monitor = {"top": 100, "left": 200, "width": 750, "height": 600}
fps = 0
display_time = 2
sct = mss.mss()
#modelio pavadinimas
MODEL_NAME = 'inference_graph'
#labeliu map
PATH_TO_LABELS = os.path.join('/home/todd/Documents/TensorFlow/models/research/object_detection/data',
                              'mscoco_label_map.pbtxt')
#nurodom modeli kuri atsisuntem
PATH_TO_FROZEN_GRAPH = os.path.join(
    '/home/todd/PycharmProjects/Deep-Learning-with-screen-cap/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb')
# Nurodom kieki klasiu ieskome
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# uzkraunam modeli
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Detectionas
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      # gaunam pixelius ir verciam juos i np array
      image_np = np.array(sct.grab(monitor))
      # uzdedam spalvas
      image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
      # keiciam dimencijas nes tensorius tikesi tokiu [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detectionas.
      #nurdom kokio detectiona mums reikia image nes tai pixeliai boxes nes norim matyt ka butent fixuoja ir kad paibreztu
      # keturkampa scores nes idomu kaip tiksliai clase nes norim zinot ka mato num_detection yra number plates detection
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # db visa ta ka mes isireiskem atiduodam tensuriauj ir paleidziam su sitais parametrais
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=3)
      # atvaizdavimas
      cv2.imshow(title, cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
      # FPS skaiciavimas kad zinotumem kaip greit sukasi
      fps+=1
      TIME = time.time() - start_time
      if TIME >= display_time:
        print("FPS: ", fps / TIME)
        fps = 0
        start_time = time.time()
      # q jei nori uzbaigt
      if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
