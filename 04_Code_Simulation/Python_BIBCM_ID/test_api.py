from openai import OpenAI

client = OpenAI(
    api_key="sk-billing-6e3710172696328ecd0124af746a1257da341d842268aad0",
    base_url="https://nexai.newdev.net/api/v1"
)

response = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "Viết hàm Python tính giai thừa."}],
    max_tokens=1024
)

print(response.choices[0].message.content)