import json
import re

def extract_json(text: str) -> dict:
    """
    Robustly extracts the first JSON object from a string that may contain yapping or extra text.
    """
    # Try finding JSON within markdown code blocks first
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        content = json_match.group(1).strip()
    else:
        # Try finding everything from the first { to the last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            content = text[start:end+1]
        else:
            content = text

    # Attempt to use JSONDecoder to handle trailing garbage if necessary
    try:
        # First attempt: simple load
        return json.loads(content)
    except json.JSONDecodeError:
        # Second attempt: Stop at the first valid object
        try:
            decoder = json.JSONDecoder()
            # If content has multiple objects or trailing text, raw_decode might help
            # We look for the first { and try from there
            start = content.find("{")
            if start != -1:
                obj, _ = decoder.raw_decode(content[start:])
                return obj
        except Exception:
            pass
            
    raise ValueError(f"Could not extract valid JSON from: {text[:100]}...")
