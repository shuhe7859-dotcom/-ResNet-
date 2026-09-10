"""
大模型融合模块 —— 对接 DeepSeek API，实现分类纠错与精判
"""

import json
import urllib.request

import config


# ==================== 本地知识库 ====================
DISPOSAL_GUIDES = {
    "cardboard": "投入蓝色可回收物桶。拆开压扁、去除胶带和订书钉，保持干燥不沾油污。",
    "glass":     "投入蓝色可回收物桶。清空内容物、简单冲洗，瓶盖与瓶身分开投放。",
    "metal":     "投入蓝色可回收物桶。倒空残留、简单冲洗，尖锐边角包好后投放。",
    "paper":     "投入蓝色可回收物桶。保持干燥、去除塑料覆膜或胶带。",
    "plastic":   "投入蓝色可回收物桶。倒空残留、简单冲洗，去除标签和瓶盖。",
    "organic":   "投入绿色厨余垃圾桶。沥干水分后投放，勿混入塑料袋或餐具。",
    "trash":     "投入灰色其他垃圾桶。无法回收的非可回收物均投入此处。",
}

ECO_TIPS = {
    "cardboard": "回收1吨废纸板可少砍17棵大树，节省填埋空间2.5立方米。",
    "glass":     "玻璃100%可无限次回收，回收一个玻璃瓶省的电可让灯泡亮4小时。",
    "metal":     "回收铝罐节省的能源高达生产新铝罐的95%，循环周期仅60天。",
    "paper":     "废纸回收再造纸可减少75%的空气污染和35%的水污染。",
    "plastic":   "塑料自然降解需要400年以上，回收1吨塑料可节省6吨石油资源。",
    "organic":   "厨余垃圾堆肥后变成有机肥料，可减少50%以上的垃圾填埋量。",
    "trash":     "正确分类可使垃圾减量60%，减少焚烧产生的二噁英排放。",
}


# ==================== 提示词构造 ====================
def build_prompt(top3_results, class_names):
    """构造通用、严谨的推理提示词"""
    top3_str = "\n".join(
        f"  {i+1}. {item['class']} (置信度: {item['confidence']:.2%})"
        for i, item in enumerate(top3_results)
    )

    allowed = ", ".join(class_names)

    prompt = f"""你是专业垃圾分类专家，严格按照国家标准推理，不要只看置信度。

已知：
1. 垃圾分类类别只能从以下选择：{allowed}
2. ResNet 给出的 Top3 候选类别：\n{top3_str}
3. 你不能看到图片，只能根据候选类别和常识推理

推理规则：
1. 先排除明显不合理、不符合常识的类别
2. 在合理类别中选择最符合日常物品的一个
3. 只能从上面 {len(class_names)} 个类别中选择，不能编造
4. 只输出一个英文单词，不要任何解释、文字、符号"""

    return prompt


# ==================== API 调用 ====================
def call_deepseek(prompt):
    """调用 DeepSeek API 进行语义精判"""
    if config.DEEPSEEK_API_KEY == "your-api-key-here":
        print("[警告] 未配置 DeepSeek API Key，跳过大模型调用，直接使用 ResNet 结果")
        return None

    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个专业、严谨的垃圾分类专家。只输出一个英文单词，不要任何解释。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 32,
        "stream": False,
    }

    req = urllib.request.Request(
        config.DEEPSEEK_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[错误] DeepSeek API 调用失败: {e}")
        return None


# ==================== 响应解析 ====================
def parse_llm_response(response_text, default_class, class_names):
    """解析大模型返回的单个英文类别词，匹配本地知识库"""
    result = {
        "final_class": default_class,
        "disposal_guide": DISPOSAL_GUIDES.get(default_class, "请查询当地垃圾分类指南。"),
        "eco_tip": ECO_TIPS.get(default_class, "垃圾分类人人有责，从你我做起。"),
    }

    if response_text is None:
        return result

    word = response_text.strip().lower().rstrip(".,;:!?。，；：！？")
    if word in class_names:
        result["final_class"] = word
    else:
        for name in class_names:
            if name in word:
                result["final_class"] = name
                break

    final = result["final_class"]
    result["disposal_guide"] = DISPOSAL_GUIDES.get(final, "请查询当地垃圾分类指南。")
    result["eco_tip"] = ECO_TIPS.get(final, "垃圾分类人人有责，从你我做起。")

    return result


# ==================== 主入口 ====================
def llm_refine(top3_results, class_names):
    prompt = build_prompt(top3_results, class_names)
    response = call_deepseek(prompt)
    default_class = top3_results[0]["class"]
    return parse_llm_response(response, default_class, class_names)
