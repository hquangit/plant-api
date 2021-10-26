from logging import debug
import multiprocessing
import os
import sys

from flask.helpers import make_response
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
from imutils.video import VideoStream
#import tensorflow.compat.v1
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import threading
from flask_swagger_ui import get_swaggerui_blueprint
# from imutils.video import VideoStream
from flask import Response
from flask import Flask, jsonify, request, stream_with_context
from flask import render_template
import argparse
import json
from bson import json_util, ObjectId
import json
from bson.json_util import dumps
import jsonpickle
from t_classes.classes import InteractiveVideo
import uuid, random
from waitress import serve
from flask_cors import CORS, cross_origin
from multiprocessing import Process
import gc
from werkzeug.serving import WSGIRequestHandler
from imutils.video import FileVideoStream
from utils.http import successResponse, failedResponse


# flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
# flags.DEFINE_string('framework', './checkpoints/yolov4-416',
#                     'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to') # 416
# flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
# flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# flags.DEFINE_string('video', './data/video/pig3.mp4', 'path to input video or set to 0 for webcam')
# flags.DEFINE_string('output', None, 'path to output video')
# flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
# flags.DEFINE_float('iou', 0.45, 'iou threshold')
# flags.DEFINE_float('score', 0.50, 'score threshold')
# flags.DEFINE_boolean('dont_show', False, 'dont show video output')
# flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
# flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

class Position:
    def __init__(self, tracking_id, min_x, min_y, max_x, max_y):
        self.tracking_id = tracking_id
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

class Identity:
    def __init__(self, id: str, pos: Position):
        self.id = id
        self.pos = pos
class FLAGS_CLASS:
    def __init__(self, framework, weights, size, tiny, model, video, output, output_format, iou, score, dont_show, info, count):
        self.framework = framework
        self.weights = weights
        self.size = size
        self.tiny = tiny
        self.model = model
        self.video = video
        self.output = output
        self.output_format = output_format
        self.iou = iou
        self.score = score
        self.namdont_show = dont_show
        self.info = info
        self.count = count

# Global variables
# outputFrame = None
# lock = threading.Lock()
# trackingId = None
clientDict = {}
#config = ConfigProto()
#session = InteractiveSession(config=config)


# initialize a flask object
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = uuid.uuid4().hex

#try:
    #vid = cv2.VideoCapture(int(video_path))

print("up commiing")
    #vid = VideoStream(src=0).start()
#except:
    #vid = cv2.VideoCapture(video_path)

def findCamera(id):
    cameras = [0, 1, './data/video/pig3.mp4', './data/video/11_nursery_high_activity_day.mp4', './data/video/test_video_1p.mp4', 'rtsp://admin:camera1460@192.168.1.30:554']
    return cameras[int(id)]

# ==========================
# comment when deploy
@app.route("/checkin/", methods=["GET"])
@cross_origin()
def checkin():
    gc.collect()
	# return the rendered template
    return successResponse(str(threading.active_count()))

@app.route("/")
@cross_origin()
def index():
	# return the rendered template
    gc.collect()
    return render_template("index.html")

def detection(id, client_uuid: str):
    # maybe show video when received post api request.
    # sess = tf.InteractiveSession()
    gc.collect()
    global clientDict, FLAGS, infer, encoder
    try:
        # load configuration for object detector
        #onfig.gpu_options.allow_growth = True
        gc.collect()
        id = 3
        imultilMOD = False
        clientDict[client_uuid] = {}
        clientDict[client_uuid]['pigId'] = '0'
        clientDict[client_uuid]['width'] = 0
        clientDict[client_uuid]['height'] = 0
        clientDict[client_uuid]['xAxis'] = 0
        clientDict[client_uuid]['yAxis'] = 0
        clientDict[client_uuid]['stop_the_thread'] = False
        cam = findCamera(id)
        print(cam)
        if imultilMOD == True:
            vid = FileVideoStream(cam).start()
        else:
            vid = cv2.VideoCapture(cam)
        print('new cam')
        #vid = VideoStream(src=cam).start()
        del id
        time.sleep(2.0)
        if imultilMOD == False:
            clientDict[client_uuid]['FRAME_WIDTH'] = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            clientDict[client_uuid]['FRAME_HEIGHT'] = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid
        print(clientDict)
        print("in detectPig")
        # global outputFrame, trackingId
        
        nms_max_overlap = 1.0

        max_cosine_distance = 0.4
        nn_budget = None
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        print('In main')
        
        #FLAGS(flags)
        # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = FLAGS.size
        #video_path = FLAGS.video
        #print('In detectPig True')
        #saved_model_loaded = None
        #infer = None

        # load tflite model if flag is set
        if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
        # otherwise load standard tensorflow saved model
        # else:
            
            

        # begin video capture
        
        if imultilMOD == False and not vid.isOpened():
            raise IOError("Cannot open webcam")

        print('In detectPig True')
        # if FLAGS.output:
        #     # by default VideoCapture returns float instead of int
        #     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     fps = int(vid.get(cv2.CAP_PROP_FPS))
        #     codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        #     out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

        # get video ready to save locally if flag is set
        # print(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)))
        # print(int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # =============== for saving output ================
        # if FLAGS.output:
        #     # by default VideoCapture returns float instead of int
        #     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     fps = int(vid.get(cv2.CAP_PROP_FPS))
        #     _fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        #     codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        #     out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

        frame_num = 0
        outputFrame = None
        # while video is running
        print("Prequense")
        try:
            while True:
                if clientDict[client_uuid]['stop_the_thread'] == True:
                    clientDict.pop(client_uuid)
                    print(clientDict)
                    gc.collect
                    break
                #if client.is_connected():
                #    stop_the_thread = True
                #if stop_the_thread:
                #    raise Exception('Client is close!')
                gc.collect()
                print("in True")
                return_value = False
                frame = None
                if imultilMOD:
                    frame = vid.read()
                else:
                    return_value, frame = vid.read()
                highlight_id = clientDict[client_uuid]['pigId']
                print('highlight ID: ')
                print(highlight_id)

                #frame = imutils.resize(frame, width=720)
                #frame = imutils.resize(frame, height=1280)
                if imultilMOD == False:
                    if return_value:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        #image = Image.fromarray(frame)
                    else:
                        print('Video has ended or failed, try a different video format!')
                        break
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #if frame.any():
                # # # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # # # image = Image.fromarray(frame)
                #else:
                    #print('Video has ended or failed, try a different video format!')
                    #break
                frame_num +=1
                print('Frame #: ', frame_num)
                frame_size = frame.shape[:2]
                print(frame_size)
                image_data = cv2.resize(frame, (input_size, input_size))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                start_time = time.time()

                # run detections on tflite if flag is set
                if FLAGS.framework == 'tflite':
                    interpreter.set_tensor(input_details[0]['index'], image_data)
                    interpreter.invoke()
                    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                    # run detections using yolov3 if flag is set
                    if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                        input_shape=tf.constant([input_size, input_size]))
                    else:
                        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                        input_shape=tf.constant([input_size, input_size]))
                else:
                    batch_data = tf.constant(image_data)
                    pred_bbox = infer(batch_data)
                    for key, value in pred_bbox.items():
                        boxes = value[:, :, 0:4]
                        pred_conf = value[:, :, 4:]

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=FLAGS.iou,
                    score_threshold=FLAGS.score
                )

                # convert data to numpy arrays and slice out unused elements
                num_objects = valid_detections.numpy()[0]
                bboxes = boxes.numpy()[0]
                bboxes = bboxes[0:int(num_objects)]
                scores = scores.numpy()[0]
                scores = scores[0:int(num_objects)]
                classes = classes.numpy()[0]
                classes = classes[0:int(num_objects)]

                # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
                original_h, original_w, _ = frame.shape
                bboxes = utils.format_boxes(bboxes, original_h, original_w)

                # store all predictions in one parameter for simplicity when calling functions
                pred_bbox = [bboxes, scores, classes, num_objects]

                # read in all class names from config
                class_names = utils.read_class_names(cfg.YOLO.CLASSES)

                # by default allow all classes in .names file
                #allowed_classes = list(class_names.values())
                
                # ============================ BIGNOTE ================================
                # custom allowed classes (uncomment line below to customize tracker for only people)
                allowed_classes = ['pig']

                # loop through objects and use class index to get class name, allow only classes in allowed_classes list
                names = []
                deleted_indx = []
                for i in range(num_objects):
                    class_indx = int(classes[i])
                    class_name = class_names[class_indx]
                    if class_name not in allowed_classes:
                        deleted_indx.append(i)
                    else:
                        names.append(class_name)
                names = np.array(names)
                count = len(names)
                if FLAGS.count:
                    cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                    print("Objects being tracked: {}".format(count))
                # delete detections that are not in allowed_classes
                bboxes = np.delete(bboxes, deleted_indx, axis=0)
                scores = np.delete(scores, deleted_indx, axis=0)

                # encode yolo detections and feed to tracker
                features = encoder(frame, bboxes)
                detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

                #initialize color map
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.class_name for d in detections])
                indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]       

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                # update tracks
                list_cord = []
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                
                # draw bbox on screen
                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    if str(track.track_id) == highlight_id:
                        cv2.putText(frame, class_name + "-" + str(track.track_id) + 'active',(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    else:
                        cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

                # if enable info flag then print details about each track
                    if FLAGS.info:
                        trackingId = str(track.track_id)
                        list_cord.append(Position(trackingId, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                        #print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

                # calculate frames per second of running detections
                clientDict[client_uuid]['lst_pig_pos'] = list_cord
                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)
                print('i: ', i)
                # result = np.asarray(frame)
                # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                #if not FLAGS.dont_show:
                    #cv2.imshow("Output Video", result)
                
                # if output flag is set, save video file
                #if FLAGS.output:
                    #out.write(result)
                #with lock:
                gc.collect()
                outputFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
                (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                    bytearray(encodedImage) + b'\r\n')
            
                #if cv2.waitKey(1) & 0xFF == ord('q'): break
            # p = Process(target=on_process, args=(vid, ))
            # p.start()
            # yield p 
            # p.join()
        except Exception as e:
            print('Exception in generator', e.__class__)
    # except Exception as e:
    #     import gc
    #     gc.collect()
    #     print("Oops!", e.__class__, "occurred.")
    #     print("Next entry.")
    #     print()
    except Exception as ex:
        print("Something went wrong")
        print("Oops!", ex.__class__, " error when streaming")
    finally:
        print('Connection is closed!!')
        print('session end!!')
        if imultilMOD:
            vid.stop()
        else:
            vid.release()
        del vid
        if client_uuid in clientDict:
            clientDict.pop(client_uuid)
            print(clientDict)
        # cv2.destroyAllWindows()
        # (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        # return(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
        #     bytearray(encodedImage) + b'\r\n')
        # del client_uuid, flag, encodedImage, outputFrame, list_cord, image_data, highlight_id, input_size
        # del frame, start_time, pred, pred_conf, batch_data, pred_bbox, boxes, scores, classes, valid_detections,
        # num_objects, bboxes, pred_bbox, class_names, allowed_classes, names, deleted_indx, 
        # class_indx, class_name, count, features, detections, indices, bbox, fps, encodedImage, frame_size
        # nms_max_overlap, frame_num
        gc.collect()
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        return(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
        # del return_value, frame, highlight_id, image_data, start_time,
        # pred, boxes, pred_conf, batch_data, scores, classes, valid_detections,
        # num_objects, bboxes, pred_bbox, class_names, allowed_classes, names, deleted_indx, 
        # class_indx, class_name, count, features, detections, list_cord, indices, bbox, fps, encodedImage, flag
        # del input_size, max_cosine_distance, nn_budget, nms_max_overlap, model_filename, encoder, metric,
        # tracker, FLAGS, interpreter, input_details, output_details, saved_model_loaded, frame_num, infer
        #sys.exit()
    # print("INFO")
    # print(id)
    # print(client_uuid)
    # p = Process(target=run_in_process, args=(id,client_uuid,))
    # print("Before Start")
    # p.start()
    # p.join()
    # gc.collect()

# def generate():
# 	#grab global references to the output frame and lock variables
# 	global outputFrame
# 	#loop over frames from the output stream
# 	while True:
#         (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
#         if not flag:
#             continue
# 		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
# 			bytearray(encodedImage) + b'\r\n')

@app.route('/videoFeed/<string:id>', methods=["GET", 'POST'])
@cross_origin()
def video_feed(id):
    # return the response generated along with the specific media
    # type (mime type)
    try:
        if request.method == 'GET':
            print('going on GET')
            client_uuid = str(uuid.uuid4())
            gc.collect()
            id = 3
            return Response(stream_with_context(detection(id, client_uuid)),
            mimetype = "multipart/x-mixed-replace; boundary=frame",
            headers={'client-uuid': client_uuid})
        else:
            print('going on POST')
            if not request.json:
                return Response(
                "The request body has somethings wrong!!",
                status=400,
            )
            results = request.json
            gc.collect()
            id = 3
            return Response(response=stream_with_context(detection(id, results['clientUuid'])),
                mimetype = "multipart/x-mixed-replace; boundary=frame",
                headers={'client-uuid': results['clientUuid']})
    except:
        return make_response(failedResponse("The request body has somethings wrong!!","BadRequest"), 400)


@app.route('/videoFeed/id/<string:id>/clientUuid/<string:clientUuid>', methods=["GET"])
@cross_origin()
def video_feed_2(id, clientUuid):
    # return the response generated along with the specific media
    # type (mime type)
    try:
        print('going on GET')
        gc.collect()
        id = 3
        return Response(stream_with_context(detection(id, clientUuid)),
            mimetype = "multipart/x-mixed-replace; boundary=frame",
            headers={'client-uuid': clientUuid})
    except:
        return make_response(failedResponse("The request body has somethings wrong!!","BadRequest"), 400)
    

@app.route('/terminate', methods=["POST"])
@cross_origin()
def terminate():
    # return the response generated along with the specific media
    # type (mime type)
    try:
        global clientDict
        if not request.json:
            return make_response(failedResponse("The request body has somethings wrong!!","BodyWrong"), 400)
        lstTerminated = []
        results = request.json
        print(results)
        for item in results:
            if item in clientDict:
                clientDict[item]['stop_the_thread'] = True
                lstTerminated.append(item)
        gc.collect()
        return successResponse(lstTerminated)
    except Exception as e:
        gc.collect()
        print("Oops!", e.__class__, " error when get video_feed")
        print("Next entry.")
        return make_response(failedResponse("Error when get video_feed","ErrorVideoFeed"), 500)

@app.route('/terminate_all', methods=["GET"])
@cross_origin()
def terminate_all():
    # return the response generated along with the specific media
    # type (mime type)
    try:
        global clientDict
        # if not request.json:
        #     return Response(
        #     "The request body has somethings wrong!!",
        #     status=400,
        # )
        lstTerminated = []
        for item in clientDict:
            print(item)
            clientDict[item]['stop_the_thread'] = True
            lstTerminated.append(item)
        gc.collect()
        return successResponse(lstTerminated)
    except Exception as e:
        gc.collect()
        print("Oops!", e.__class__, " error when get video_feed")
        print("Next entry.")
        return make_response(failedResponse("Error when get video_feed","ErrorVideoFeed"), 500)

@app.route("/position/<string:client_uuid>/", methods=["GET"])
@cross_origin()
def position(client_uuid):
    try:
        results = []
        lst_pig_pos = clientDict[client_uuid]['lst_pig_pos']
        for box in lst_pig_pos:
            tmp = {}
            tmp['pigId'] = box.tracking_id
            tmp['min_x'] = box.min_x
            tmp['min_y'] = box.min_y
            tmp['max_x'] = box.max_x
            tmp['max_y'] = box.max_y
            # results.append(json.dumps(box.__dict__))
            results.append(tmp)
            print("BOXXXXXX: ")
            print(box)
            gc.collect()
        return make_response(successResponse(results))
    except Exception as e:
        gc.collect()
        print("Oops!", e.__class__, " error when get all position of pigs")
        print("Next entry.")
        return make_response(failedResponse("Error when get all position of pigs", "Exception"),500)
    #return jsonify(lstBox)

@app.route("/highlightID", methods=["POST"])
@cross_origin()
def highlight_id():
    try:
        global clientDict
        print(request.json)
        if request == None:
            return make_response(failedResponse("The response body wrong!", "BodyWrong"),400)
        results = request.json
        print(results)
        if results['client_uuid'] not in clientDict:
            return make_response(failedResponse("Wrong client UUID", 'WrongUUID'),400)
        clientDict[str(results['client_uuid'])]['pigId'] = results['pigId']
        gc.collect()
        return make_response(successResponse("Highlight successfully!"), 200)
    except Exception as e:
        print("Oops!", e.__class__, " error when execute highlightID API")
        print("Next entry.")
        return make_response(failedResponse("Error when execute api calculate highlight ID", "Exception"),500)

def calculate_new_pivot(oldValue, oldWH, newWH):
    return float(oldValue*newWH/oldWH)

def calculate_new_position(xAxis, yAxis, width, height, client_uuid):
    try:
        global clientDict
        x_new = calculate_new_pivot(xAxis, width, clientDict[client_uuid]['FRAME_WIDTH'])
        y_new = calculate_new_pivot(yAxis, height, clientDict[client_uuid]['FRAME_HEIGHT'])
        lst_pig_pos = clientDict[client_uuid]['lst_pig_pos']
        for item in lst_pig_pos:
            if x_new >= item.min_x and x_new <= item.max_x and y_new >= item.min_y and y_new <= item.max_y:
                clientDict[client_uuid]['pigId'] = item.tracking_id
                return item.tracking_id
        return 0
    except Exception as e:
        gc.collect()
        print("Oops!", e.__class__, " error when calculate_new_position")
        print("Next entry.")
        return 0
    

@app.route("/highlightPosition", methods=["POST"])
@cross_origin()
def highlight_position():
    try:
        global clientDict
        print(request.json)
        if request == None:
            return make_response(failedResponse("The request body has somethings wrong!!","BadRequest"), 400)
        results = request.json
        print(request.json)
        if str(results['client_uuid']) not in clientDict:
            return make_response(failedResponse("Wrong client UUID!!","WrongUUID"), 400)
        results = request.json
        clientDict[f"{results['client_uuid']}"]['xAxis'] = results['x']
        clientDict[f"{results['client_uuid']}"]['yAxis'] = results['y']
        clientDict[f"{results['client_uuid']}"]['width'] = results['width']
        clientDict[f"{results['client_uuid']}"]['height'] = results['height']
        res = calculate_new_position(results['x'], results['y'], results['width'], results['height'], results['client_uuid'])
        gc.collect()
        if res != 0:
            tmpRes = {}
            tmpRes['result'] = True
            tmpRes['message'] = 'Highlight successfully'
            tmpRes['pigId'] = res
            return make_response(successResponse(tmpRes), 200)
        else:
            return make_response(failedResponse("Can't find pig's position","PigNotFound"), 404)
    except Exception as e:
        gc.collect()
        print("Oops!", e.__class__, " error when execute api calculate highlight location")
        print("Next entry.")
        return make_response(failedResponse("Error when execute api calculate highlight location", "Exception"),500)
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--ip", type=str, required=True,
	# 	help="ip address of the device")
	# ap.add_argument("-o", "--port", type=int, required=True,
	# 	help="ephemeral port number of the server (1024 to 65535)")
	# ap.add_argument("-f", "--frame-count", type=int, default=32,
	# 	help="# of frames used to construct the background model")
	#args = vars(ap.parse_args())
    #
    gc.enable()
    FLAGS = FLAGS_CLASS('tf', './checkpoints/yolov4-416', 416, False, 'yolov4', './data/video/pig3.mp4', None, 'XVID', 0.45, 0.50, False, True, False)
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    # Definition of the parameters
    # initialize deep sort
    model_filename = 'model_data/mars.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    print('load successfully')
    #app.run('0.0.0.0', 8000, debug=False, threaded=True, use_reloader=False)
    #app.run(debug=True)
    #kwargs = {'host': '0.0.0.0', 'port': 8000, 'threaded': True, 'use_reloader': False, 'debug': False}

    #   running flask thread
    #flaskThread = threading.Thread(target=app.run, daemon=True, kwargs=kwargs).start()
    # ============================== BIGNOTE ========================
	# start a thread that will perform motion detection
    # start right after run server
	#t = threading.Thread(target=detection, args=(
	#	args["frame_count"],))
	#t.daemon = True
	#t.start()
    # ============================== BIGNOTE ========================
	# start the flask app
    try:
        from waitress import serve
        WSGIRequestHandler.protocol_version = "HTTP/1.1"
        serve(app, host='0.0.0.0', port=8000, threads=1000)
    except Exception as e:
       print("Oops!", e.__class__, " error when start serve a connection")
       print("Next entry.")
	#serve()
    
        
cv2.destroyAllWindows()
# release the video stream pointer
#vid.stop()