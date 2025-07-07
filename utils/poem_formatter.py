import re

def format_poem(tokens_or_text, poem_type='五言'):
    # 1. 转成字符串
    if isinstance(tokens_or_text, list):
        text = "".join(tokens_or_text)
    else:
        text = tokens_or_text

    # 2. 只保留汉字，去掉其他符号
    chars = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # 3. 限制最大字数（40字或56字）
    max_chars = 40 if poem_type == '五言' else 56
    chars = chars[:max_chars]

    # 4. 句子格式化
    char_per_line = 5 if poem_type == '五言' else 7

    lines = []
    for i in range(0, len(chars), char_per_line):
        line = chars[i:i+char_per_line]
        if not line:
            continue
        # 5. 添加标点
        if (i // char_per_line) % 2 == 0:
            line += '，'
        else:
            line += '。\n'
        lines.append(line)

    return "".join(lines).strip()
