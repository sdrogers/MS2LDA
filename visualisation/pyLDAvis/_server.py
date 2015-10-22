# this file is largely based on https://github.com/jakevdp/mpld3/blob/master/mpld3/_server.py
# Copyright (c) 2013, Jake Vanderplas
"""
A Simple server used to serve LDAvis visualizations
"""
import itertools
import random
import socket
import sys
import threading
from urllib import urlopen, pathname2url
import urlparse
import webbrowser
import StringIO
from . import urls
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from ..networkx import lda_visualisation
import json

IPYTHON_WARNING = """
Note: if you're in the IPython notebook, pyLDAvis.show() is not the best command
      to use. Consider using pyLDAvis.display(), or pyLDAvis.enable_notebook().
      See more information at http://pyLDAvis.github.io/quickstart.html .

You must interrupt the kernel to end this command
"""

try:
    # Python 2.x
    import BaseHTTPServer as server
except ImportError:
    # Python 3.x
    from http import server

class GlobalVariable(object):
    selected_topic_id = 0
    ms1_idx = 0   
    degree = 0 

# for http://bugs.python.org/issue6193
def get_url_path(relative_path):
    abs_path = os.path.abspath(relative_path)
    url = pathname2url(abs_path)
    return urlparse.urljoin('file:', url)    

def generate_handler(html, files=None, topic_plotter=None):
    
    if files is None:
        files = {}
    
    logo_url = get_url_path(urls.DEFAULT_LOGO_LOCAL)
    show_graph_url = get_url_path(urls.DEFAULT_SHOW_GRAPH_LOCAL)
    print "logo_url is " + logo_url
    print "show_graph_url is " + show_graph_url
    
    # add default images to files
    logo_content = StringIO.StringIO(urlopen(logo_url).read()).read()
    show_graph_content = StringIO.StringIO(urlopen(show_graph_url).read()).read()
    files['/images/default_logo.png'] = ('image/png', logo_content)
    files['/images/graph_example.jpg'] = ('image/jpg', show_graph_content)

    class MyHandler(server.BaseHTTPRequestHandler):

        def do_GET(self):
            """Respond to a GET request."""

            # serve main document        
            if self.path == '/':
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write("<html><head>"
                                 "<title>LDAvis</title>"
                                 "</head><body>\n".encode())
                self.wfile.write(html.encode())
                self.wfile.write("</body></html>".encode())

            # handle request for MS1 plots
            elif self.path.startswith('/topic'):

                # get everything after '?'                    
                path, tmp = self.path.split('?', 1)
                qs = urlparse.parse_qs(tmp)
                action = qs['action'][0]
                
                if action == 'set':
                    # keep track of the current topic that has been clicked
                    circle_id = qs['circle_id'][0]
                    topic_id = self._get_topic_id(circle_id)                    
                    GlobalVariable.ms1_idx = 0
                    GlobalVariable.selected_topic_id = topic_id
                elif action == 'load':
                    # return first ms1 plot in the topic while hovering
                    circle_id = qs['circle_id'][0]
                    topic_id = self._get_topic_id(circle_id)
                    GlobalVariable.ms1_idx = 0
                elif action == 'next':
                    topic_id = GlobalVariable.selected_topic_id
                    if topic_id in topic_plotter.topic_ms1_count:
                        max_count = topic_plotter.topic_ms1_count[topic_id]
                        if (GlobalVariable.ms1_idx + 1) < max_count:
                            GlobalVariable.ms1_idx += 1
                elif action == 'prev':
                    topic_id = GlobalVariable.selected_topic_id
                    if (GlobalVariable.ms1_idx - 1) >= 0:
                        GlobalVariable.ms1_idx -= 1
                elif action == 'show':
                    topic_id = GlobalVariable.selected_topic_id                    

                # get the image content
                fig = topic_plotter.plot_for_web(topic_id, GlobalVariable.ms1_idx)
                if fig is not None:
                    # topic has some ms1 plot
                    canvas = FigureCanvas(fig)
                    output = StringIO.StringIO()
                    canvas.print_png(output)  
                    content = output.getvalue()
                    content_type = 'image/png'
                else:
                    # topic has no ms1 plots
                    content_type, content = files['/images/default_logo.png']

                # send image content to response
                self.send_response(200)
                self.send_header("Content-type", content_type)
                self.end_headers()
                self.wfile.write(content)      

            # handle request for the clickable graph
            elif self.path.startswith('/graph.html'):

                # get everything after '?'                    
                path, tmp = self.path.split('?', 1)
                qs = urlparse.parse_qs(tmp)
                degree = qs['degree'][0]
                GlobalVariable.degree = int(degree)
                
                content_type, content = files[path]
                self.send_response(200)
                self.send_header("Content-type", content_type)
                self.end_headers()
                self.wfile.write(content.encode())
                                
            # handle request for the clickable graph
            elif self.path.startswith('/graph.json'):
                
                print "Serving dynamic json file -- threshold = " + str(GlobalVariable.degree)
                print "to_highlight = " + str(topic_plotter.to_highlight)
                json_data, G = lda_visualisation.get_json_from_docdf(topic_plotter.docdf.transpose(), 
                                                                  topic_plotter.to_highlight,
                                                                  GlobalVariable.degree)

#                 print "Debugging file saved to " + json_outfile
#                 json_outfile = '/home/joewandy/git/metabolomics_tools/justin/visualisation/pyLDAvis/json_out.json'
#                 with open(json_outfile, 'w') as f:
#                     json.dump(json_data, f, sort_keys=True, indent=4, ensure_ascii=False)                
                
                content_type = "application/json"
                content = json.dumps(json_data, sort_keys=True, indent=4, ensure_ascii=False)                
                self.send_response(200)
                self.send_header("Content-type", content_type)
                self.end_headers()
                self.wfile.write(content.encode())                          
               
            elif self.path in files:
                                  
                # handle other images request, serve the content without encode()
                if self.path.startswith('/images'):
                    content_type, content = files[self.path]
                    self.send_response(200)
                    self.send_header("Content-type", content_type)
                    self.end_headers()
                    self.wfile.write(content)
                    
                # serve any other content
                else:
                    content_type, content = files[self.path]
                    self.send_response(200)
                    self.send_header("Content-type", content_type)
                    self.end_headers()
                    self.wfile.write(content.encode())

            else:
                self.send_error(404)

        def _get_topic_id(self, circle_id):
            # circle_id will look like this: 'ldavis_el55781404281730576168333350176-topic2' for topic 1
            tokens = circle_id.split('-')
            topic_str = tokens[1] # get e.g. 'topic2'
            topic_id = topic_str[5:] # get rid of the front 'topic' bit from topic_str
            topic_id = int(topic_id)-1 # since on the javascript side, we index from 1,.. internally
            return topic_id

    return MyHandler


def find_open_port(ip, port, n=50):
    """Find an open port near the specified port"""
    ports = itertools.chain((port + i for i in range(n)),
                            (port + random.randint(-2 * n, 2 * n)))

    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((ip, port))
        s.close()
        if result != 0:
            return port
    raise ValueError("no open ports found")


def serve(html, ip='127.0.0.1', port=8888, n_retries=50, files=None,
          ipython_warning=True, open_browser=True, http_server=None,
          topic_plotter=None):
    """Start a server serving the given HTML, and (optionally) open a
    browser

    Parameters
    ----------
    html : string
        HTML to serve
    ip : string (default = '127.0.0.1')
        ip address at which the HTML will be served.
    port : int (default = 8888)
        the port at which to serve the HTML
    n_retries : int (default = 50)
        the number of nearby ports to search if the specified port is in use.
    files : dictionary (optional)
        dictionary of extra content to serve
    ipython_warning : bool (optional)
        if True (default), then print a warning if this is used within IPython
    open_browser : bool (optional)
        if True (default), then open a web browser to the given HTML
    http_server : class (optional)
        optionally specify an HTTPServer class to use for showing the
        figure. The default is Python's basic HTTPServer.
    """
    port = find_open_port(ip, port, n_retries)
    Handler = generate_handler(html, files, topic_plotter)

    if http_server is None:
        srvr = server.HTTPServer((ip, port), Handler)
    else:
        srvr = http_server((ip, port), Handler)

    if ipython_warning:
        try:
            __IPYTHON__
        except:
            pass
        else:
            print(IPYTHON_WARNING)

    # Start the server
    print("Serving to http://{0}:{1}/    [Ctrl-C to exit]".format(ip, port))
    sys.stdout.flush()

    if open_browser:
        # Use a thread to open a web browser pointing to the server
        b = lambda: webbrowser.open('http://{0}:{1}'.format(ip, port))
        threading.Thread(target=b).start()

    try:
        srvr.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        print("\nstopping Server...")

    srvr.server_close()
