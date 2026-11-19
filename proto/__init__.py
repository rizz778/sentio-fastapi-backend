# Package init for proto
import sys
from pathlib import Path

# Fix relative import when protoc generates `import sentio_gateway_pb2 as ...`
proto_dir = str(Path(__file__).parent.resolve())
if proto_dir not in sys.path:
    sys.path.insert(0, proto_dir)
