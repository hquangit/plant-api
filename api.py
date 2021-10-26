from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request, make_response, jsonify
from flask import Response
import numpy as np
import cv2
import base64

app = Flask(__name__)
# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

yolo_label_path = "yolo.txt"
weights = 'yolov4.weights'
config = 'yolov4-custom.cfg'

def successResponse(dataZip):
    return jsonify(data=dataZip, statusCode='OK')

def failedResponse(message: str, statusCode: str):
    return jsonify(message=message, statusCode=statusCode)

def convert_base64_to_image(image_base64):
    try:
        image_base64 = np.fromstring(base64.b64decode(image_base64), dtype=np.uint8)
        image_base64 = cv2.imdecode(image_base64, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return image_base64

def convert_image_to_base64(image):
    b64_string = base64.b64encode(image.read())
    return b64_string.decode('utf-8')

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    classes = None
    with open(yolo_label_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def dectect_image(image):
    Width = image.shape[1]
    Height = image.shape[0]

    class_ids = []
    confidences = []
    boxes = []

    scale = 0.00392
    net = cv2.dnn.readNet(weights, config)
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    for out in outs:
        # print(out)
        for detection in out:
            # print(detection)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    conf_threshold = 0.5
    nms_threshold = 0.4
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    return image

# with open('test.jpg', 'rb') as file:
#     base64_encode = convert_image_to_base64(file)
#     print(type(file))

# base64_decode = convert_base64_to_image(base64_encode)
# img = dectect_image(base64_decode)
#
# window_name = 'image'
# resized = cv2.resize(img, (980,980), interpolation = cv2.INTER_AREA)
# cv2.imshow(window_name,resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

@app.route('/dectect', methods=['POST'] )
@cross_origin(origin='*')
def dectect_image_process():
    try:
        if request.method == 'POST':
        # Đọc ảnh từ client gửi lên
            print(request)
            facebase64 = request.json
            # print(facebase64)
            image = convert_base64_to_image(facebase64['facebase64'])
            dect_image = dectect_image(image)

            convert_img = base64.b64encode(dect_image)

            return jsonify({"facebase64": convert_img})
        else:
            if not request.json:
                return Response(
                    "The request body has somethings wrong!!",
                    status=400,
                )
    except:
        return make_response(failedResponse("The request body has somethings wrong!!","BadRequest"), 400)

# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')



