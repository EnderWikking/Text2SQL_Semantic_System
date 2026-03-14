import os
from openai import OpenAI

# 强制走阿里的兼容模式，彻底告别 dashscope 的报错！
client = OpenAI(
    api_key="sk-90a75b57806f4794bdbb96273df856a3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def test_qwen_api():
    print("🚀 正在使用 qwen-plus 模型进行测试...")
    try:
        completion = client.chat.completions.create(
            model="qwen3.5-plus",  # 消耗你的免费额度
            messages=[
                {"role": "system", "content": "你是一个资深的数据库专家。"},
                {
                    "role": "user",
                    "content": "我在加州学校数据库里看到一个字段叫 'CDSCode'，它的样本是 '01100170109835'，它代表什么意思？",
                },
            ],
            timeout=30,
        )
        print("✅ API 链路打通，成功获取回复！\n")
        print(completion.choices[0].message.content)

    except Exception as e:
        print(f"❌ 请求失败: {e}")


if __name__ == "__main__":
    test_qwen_api()
