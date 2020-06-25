from Web import web_start, app

print("The web server has been started at http://127.0.0.1:5000. Press <Ctrl-C> to close")

try:
    web_start.socket_io.run(app, host='0.0.0.0')
except KeyboardInterrupt:
    web_start.socket_io.stop()
