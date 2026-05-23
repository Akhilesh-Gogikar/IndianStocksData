import sys
import time
import subprocess
import socket
from playwright.sync_api import sync_playwright

PORT = 4175
SERVER_URL = f"http://127.0.0.1:{PORT}"
APP_URL = f"{SERVER_URL}/src/index.html"

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_server():
    if is_port_in_use(PORT):
        return None
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
        cwd="/Users/akhileshgogikar/IndianStocksData/cerebral-insights-platform",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(1.0)
    return process

def main():
    server = start_server()
    print("server started.", flush=True)
    
    with sync_playwright() as pl:
        browser = pl.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        page.on("console", lambda msg: print(f"[Browser Console] {msg.type}: {msg.text}", flush=True))
        page.on("pageerror", lambda err: print(f"[Browser Uncaught Error] {err}", flush=True))
        page.on("request", lambda req: print(f"[Network Request] {req.method} {req.url}", flush=True))
        page.on("response", lambda res: print(f"[Network Response] {res.status} {res.url}", flush=True))
        
        print(f"Navigating to {APP_URL}...", flush=True)
        page.goto(APP_URL)
        time.sleep(5.0)
        
        browser.close()
    
    if server:
        server.terminate()
        server.wait()
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
