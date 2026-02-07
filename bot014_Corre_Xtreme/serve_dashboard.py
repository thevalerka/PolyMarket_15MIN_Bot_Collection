#!/usr/bin/env python3
"""
Simple HTTP server for the pair trading dashboard
Runs on port 8888 to avoid conflicts with other services
"""

import http.server
import socketserver
import os
from pathlib import Path

# Set the directory to serve
os.chdir('/home/ubuntu/013_2025_polymarket/bot014_Corre_Xtreme')

PORT = 8888

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add headers to prevent caching
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Only log errors, not every request
        if args[1][0] != '2':  # Not a 2xx response
            super().log_message(format, *args)

Handler = MyHTTPRequestHandler

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("="*60)
        print(f"Pair Trading Dashboard Server")
        print("="*60)
        print(f"Server running on port {PORT}")
        print(f"Open in browser: http://localhost:{PORT}/dashboard_standalone.html")
        print(f"Or from remote: http://YOUR_SERVER_IP:{PORT}/dashboard_standalone.html")
        print(f"Serving directory: {os.getcwd()}")
        print("="*60)
        print("Press Ctrl+C to stop")
        print()
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
except OSError as e:
    if e.errno == 98:  # Address already in use
        print(f"\nError: Port {PORT} is already in use.")
        print("Try a different port or stop the service using that port.")
    else:
        raise
