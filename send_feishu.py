# -*- coding: utf-8 -*-
"""发送飞书消息通知"""
import sys
import json
import urllib.request

# 强制 UTF-8 输出，避免 Windows 终端乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/fe3026f8-9ec1-4624-a128-cf6518499775"


def send_message(title, content, method_name="", action=""):
    """发送飞书文本消息"""
    if action:
        text = f"【{action}】{method_name}\n{title}\n{content}"
    else:
        text = f"{title}\n{content}"

    payload = {
        "msg_type": "text",
        "content": {
            "text": text
        }
    }

    headers = {
        "Content-Type": "application/json; charset=utf-8"
    }

    req = urllib.request.Request(
        WEBHOOK_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = resp.read().decode("utf-8")
            print(f"[Feishu] OK | method={method_name} action={action}")
            return True
    except Exception as e:
        print(f"[Feishu] FAIL | method={method_name} action={action} | error={e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        action = sys.argv[1]
        method_name = sys.argv[2]
        title = sys.argv[3] if len(sys.argv) > 3 else ""
        content = sys.argv[4] if len(sys.argv) > 4 else ""
        send_message(title, content, method_name, action)
    else:
        print("Usage: python send_feishu.py <action> <method_name> [title] [content]")
