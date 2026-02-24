import google.generativeai as genai
genai.configure(api_key="AIzaSyDv2ECWWZRvKVmz64Yyt9zM6OD8YZhRZeQ")

models = genai.list_models()

for m in models:
    print(m.name)
    