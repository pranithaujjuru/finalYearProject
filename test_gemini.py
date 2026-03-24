import os
from dotenv import load_dotenv
import google.generativeai as genai

def test():
    # Load env
    load_dotenv('.env.local')
    api_key = os.environ.get('GEMINI_API_KEY')
    
    # Prune it to make sure it's the new key and not the old one ending in 627
    print(f"API Key found. First 5 chars: {api_key[:5] if api_key else 'None'}, Last 5 chars: {api_key[-5:] if api_key else 'None'}")
    
    # Configure and test
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content('Say exactly "Hello World"')
        print("\nAPI Call Success! Response:")
        print(response.text)
    except Exception as e:
        print(f"\nAPI Call Failed: {e}")

if __name__ == '__main__':
    test()
