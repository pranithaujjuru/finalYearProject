import asyncio
import sys
import os

# Add the project root to sys.path to allow importing from python_agents
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from core.vision_agent import analyze_mri
from core.schemas import VisionFinding

async def main():
    print("Testing Vision Agent...")
    image_path = os.path.join(os.path.dirname(__file__), 'test_mri.jpg')
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found at {image_path}")
        return

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    try:
        result = await analyze_mri(image_bytes, "image/jpeg")
        print("\n--- Vision Result ---")
        print(f"Finding: {result.finding}")
        print(f"Severity: {result.severity}")
        print(f"Location: {result.location}")
        print("---------------------\n")
        
        if isinstance(result, VisionFinding):
            print("✅ Vision Agent test passed (Type check)")
        else:
            print("❌ Vision Agent test failed (Type check)")
    except Exception as e:
        print(f"❌ Vision Agent test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
