from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)


@socketio.on('fuck')
def handle_my_custom_event(ms):
    f = open('data.json')
    data_file = json.load(f)

    f2 = open('training.json')
    data_file2 = json.load(f2)
    print('Ay hello')
    socketio.emit('something', data_file)
    socketio.emit('something2', data_file2)


@app.route('/send', methods=['POST'])
def send():
    data = request.json['data']
    title = request.json['title']
    training = request.json['training']

    if training:
        f = open('training.json')
        data_file = json.load(f)
        data_file['data'].append(data)

        with open('training.json', 'w') as f:
            json.dump(data_file, f)

        socketio.emit('something2', request.json)
    else:
        f3 = open('data.json')
        data_file = json.load(f3)
        data_file['data'].append(data)
        data_file['title'] = title

        with open('data.json', 'w') as f3:
            json.dump(data_file, f3)

        socketio.emit('something', request.json)
    return 'OK'



if __name__ == "__main__":
    socketio.run(app, debug=True)
