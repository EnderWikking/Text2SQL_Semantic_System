import os
from openai import OpenAI

# 强制走阿里的兼容模式
client = OpenAI(
    api_key="sk-90a75b57806f4794bdbb96273df856a3",  # ⚠️ 记得填回你的 Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def test_qwen_streaming():
    print("🚀 正在召唤 qwen3.5-plus (免费版) 进行流式测试...")
    print("-" * 50)

    try:
        response = client.chat.completions.create(
            model="qwen3.5-plus",  # ✅ 认准你的 100万 免费额度专属 ID
            messages=[
                {"role": "system", "content": "你是一个资深的数据库专家。"},
                {
                    "role": "user",
                    "content": "我在加州学校数据库里看到一个字段叫 'CDSCode'，它的样本是 '01100170109835'，它代表什么意思？请简短回答。",
                },
            ],
            stream=True,  # 🌟 开启流式输出的核心开关
            timeout=30,  # 国内网络 30 秒绝对够用了
        )

        # 像打字机一样实时捕获并打印大模型的输出
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                # flush=True 保证一接收到字符就立刻印在屏幕上，绝不缓存
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n" + "-" * 50)
        print("✅ 流式测试圆满成功！免费通道完全畅通！")

    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        print(
            "💡 小提示：如果你现在开着系统级代理（VPN/梯子），建议暂时关掉再试一次，有时候代理会拦截本地的 API 请求导致超时。"
        )


if __name__ == "__main__":
    test_qwen_streaming()
