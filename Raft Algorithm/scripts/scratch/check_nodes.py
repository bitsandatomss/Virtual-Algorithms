import urllib.request
import json

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/api/snapshot") as response:
        data = json.loads(response.read().decode())
        nodes = data.get("nodes", [])
        print(f"Number of nodes: {len(nodes)}")
        if nodes:
            print(f"First node ID: {nodes[0].get('id')}")
            print(f"Last node ID: {nodes[-1].get('id')}")
except Exception as e:
    print(f"Error: {e}")
