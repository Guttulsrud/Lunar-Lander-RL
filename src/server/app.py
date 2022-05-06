from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)  # This will enable CORS for all routes

"""Socket.IO decorator to create a websocket event handler"""


@socketio.on('fuck')
def handle_my_custom_event(ms, methods=['GET', 'POST']):
    f = open('data.json')
    data_file = json.load(f)

    socketio.emit('something', data_file)

    # socketio.emit('my response', json, callback=messageReceived)


@app.route('/send', methods=['POST'])
def send():
    data = request.json['data']
    title = request.json['title']

    f = open('data.json')
    data_file = json.load(f)
    data_file['data'].append(data)
    data_file['title'] = title

    with open('data.json', 'w') as f:
        json.dump(data_file, f)

    socketio.emit('something', request.json)
    return 'OK'


if __name__ == "__main__":
    socketio.run(app, debug=True)
