#!/usr/bin/env python3
"""Sync Chinese benchmark metadata and test cases from English canonical files.

The script keeps the existing Chinese prompt, copies English canonical metadata
and test-case structure, and patches language-sensitive text checks to accept
both English and Chinese UI text.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


FILES = (
    "apps_tools.jsonl",
    "content_marketing.jsonl",
    "data_visualization.jsonl",
    "games_simulations.jsonl",
    "three_d_webgl.jsonl",
    "visual_art_animation.jsonl",
)

META_FIELDS = ("category", "source_category", "sub_type", "difficulty", "has_interaction")

# Keep this glossary deliberately conservative: it is only for assertion
# compatibility, not prompt translation.
GLOSSARY: dict[str, list[str]] = {
    "reading time": ["阅读时间", "阅读时长"],
    "min read": ["分钟阅读", "阅读时长"],
    "author": ["作者"],
    "date": ["日期", "时间"],
    "read": ["阅读"],
    "comment": ["评论", "留言"],
    "comments": ["评论", "留言"],
    "share": ["分享"],
    "tag": ["标签"],
    "tags": ["标签"],
    "previous": ["上一篇", "上一步", "上一个"],
    "prev": ["上一篇", "上一步", "上一个"],
    "next": ["下一篇", "下一步", "下一个"],
    "back": ["返回", "后退", "上一步"],
    "continue": ["继续", "下一步"],
    "finish": ["完成", "结束"],
    "turn off": ["关闭"],
    "all lights off": ["全部关灯", "关闭所有灯", "关闭所有"],
    "lights off": ["关灯", "关闭灯光"],
    "confirm": ["确认"],
    "submit": ["提交", "确认"],
    "save": ["保存"],
    "cancel": ["取消"],
    "close": ["关闭"],
    "open": ["打开"],
    "show": ["显示"],
    "hide": ["隐藏"],
    "start": ["开始", "启动"],
    "begin": ["开始", "启动"],
    "restart": ["重新开始", "重启"],
    "play": ["开始", "播放"],
    "pause": ["暂停"],
    "stop": ["停止"],
    "reset": ["重置"],
    "clear": ["清空"],
    "add to cart": ["加入购物车"],
    "add": ["添加", "新增"],
    "new": ["新建", "新增"],
    "create": ["创建", "新建"],
    "delete": ["删除", "移除"],
    "remove": ["删除", "移除"],
    "edit": ["编辑"],
    "login": ["登录", "登入"],
    "register": ["注册"],
    "sign up": ["注册"],
    "signup": ["注册"],
    "settings": ["设置"],
    "menu": ["菜单"],
    "search": ["搜索", "查找"],
    "filter": ["筛选", "过滤"],
    "sort": ["排序"],
    "export": ["导出"],
    "download": ["下载"],
    "print": ["打印"],
    "copy": ["复制"],
    "like": ["点赞", "喜欢"],
    "favorite": ["收藏", "喜欢"],
    "heart": ["收藏", "喜欢"],
    "helpful": ["有帮助"],
    "yes": ["是", "有用"],
    "no": ["否", "无用"],
    "buy": ["购买"],
    "order": ["下单", "订单"],
    "cart": ["购物车"],
    "checkout": ["结账", "结算"],
    "reserve": ["预订", "预约"],
    "reservation": ["预订", "预约"],
    "book": ["预订", "预约"],
    "ticket": ["门票", "票"],
    "rsvp": ["报名", "预约"],
    "contact sales": ["联系销售"],
    "contact": ["联系"],
    "get in touch": ["联系", "联系我们"],
    "send": ["发送", "提交"],
    "reply": ["回复"],
    "message": ["消息", "留言"],
    "chat": ["聊天", "对话"],
    "admin": ["管理", "管理员"],
    "dashboard": ["仪表板", "看板"],
    "analytics": ["分析", "统计"],
    "overview": ["概览", "总览"],
    "details": ["详情", "详细信息"],
    "detail": ["详情"],
    "gallery": ["图库", "画廊"],
    "features": ["功能", "特性"],
    "feature": ["功能", "特性"],
    "demo": ["演示"],
    "guide": ["指南"],
    "tutorial": ["教程"],
    "faq": ["常见问题", "问答"],
    "question": ["问题"],
    "general": ["通用", "常规"],
    "billing": ["账单", "计费"],
    "technical": ["技术"],
    "introduction": ["介绍", "简介"],
    "configuration": ["配置"],
    "authentication": ["认证", "身份验证"],
    "auth": ["认证", "身份验证"],
    "endpoint": ["接口", "端点"],
    "endpoints": ["接口", "端点"],
    "request": ["请求"],
    "response": ["响应"],
    "reference": ["参考", "文档"],
    "examples": ["示例", "例子"],
    "errors": ["错误"],
    "warning": ["警告"],
    "success": ["成功"],
    "info": ["信息", "提示"],
    "required": ["必填", "必需"],
    "invalid": ["无效", "错误"],
    "no results": ["无结果", "未找到"],
    "not found": ["未找到", "找不到"],
    "score": ["分数", "得分"],
    "level": ["关卡", "等级"],
    "time": ["时间"],
    "timer": ["计时器", "倒计时"],
    "health": ["生命值", "血量"],
    "lives": ["生命", "生命数"],
    "life": ["生命"],
    "coins": ["金币"],
    "coin": ["金币"],
    "stars": ["星星"],
    "star": ["星星"],
    "speed": ["速度"],
    "wave": ["波次", "关卡"],
    "move": ["移动"],
    "turn": ["回合", "轮到"],
    "winner": ["胜者", "获胜"],
    "win": ["胜利", "获胜"],
    "black": ["黑色", "黑棋"],
    "white": ["白色", "白棋"],
    "scoreboard": ["计分板"],
    "revenue": ["收入", "营收"],
    "orders": ["订单"],
    "customers": ["客户", "顾客"],
    "conversion": ["转化率"],
    "sales": ["销售", "销量"],
    "price": ["价格"],
    "pricing": ["价格", "定价"],
    "subtotal": ["小计"],
    "total": ["总计", "合计"],
    "quantity": ["数量"],
    "qty": ["数量"],
    "stock": ["股票", "库存"],
    "stocks": ["股票", "库存"],
    "bond": ["债券"],
    "bonds": ["债券"],
    "cash": ["现金"],
    "crypto": ["加密货币", "数字货币"],
    "real estate": ["房地产", "不动产"],
    "fund": ["基金"],
    "portfolio": ["投资组合"],
    "holding": ["持仓"],
    "holdings": ["持仓"],
    "allocation": ["配置", "分配"],
    "risk": ["风险"],
    "return": ["收益", "回报"],
    "benchmark": ["基准"],
    "dividend": ["股息", "分红"],
    "income": ["收入"],
    "category": ["分类", "类别"],
    "product": ["产品"],
    "products": ["产品"],
    "humidity": ["湿度"],
    "wind": ["风速"],
    "temperature": ["温度"],
    "weather": ["天气"],
    "goal": ["目标"],
    "progress": ["进度"],
    "water": ["水", "饮水"],
    "history": ["历史", "记录"],
    "today": ["今天"],
    "tomorrow": ["明天"],
    "spring": ["春", "春季"],
    "summer": ["夏", "夏季"],
    "autumn": ["秋", "秋季"],
    "fall": ["秋", "秋季"],
    "winter": ["冬", "冬季"],
    "monthly": ["月度", "每月"],
    "annual": ["年度", "年付"],
    "yearly": ["年度", "年付"],
    "month": ["月"],
    "year": ["年"],
    "light": ["浅色", "明亮", "灯", "灯光"],
    "lights": ["灯", "灯光"],
    "dark": ["暗色", "深色"],
    "theme": ["主题"],
    "jan": ["一月", "1月"],
    "feb": ["二月", "2月"],
    "mar": ["三月", "3月"],
    "apr": ["四月", "4月"],
    "may": ["五月", "5月"],
    "jun": ["六月", "6月"],
    "jul": ["七月", "7月"],
    "aug": ["八月", "8月"],
    "sep": ["九月", "9月"],
    "oct": ["十月", "10月"],
    "nov": ["十一月", "11月"],
    "dec": ["十二月", "12月"],
    "mon": ["周一", "星期一"],
    "tue": ["周二", "星期二"],
    "wed": ["周三", "星期三"],
    "thu": ["周四", "星期四"],
    "fri": ["周五", "星期五"],
    "sat": ["周六", "星期六"],
    "sun": ["周日", "星期日"],
}

TEXT_READ_MARKERS = (
    "innerText",
    "textContent",
    "aria-label",
    "placeholder",
    "document.body.innerText",
    "document.body.textContent",
)

SOURCE_READ_MARKERS = (
    "innerHTML",
    "outerHTML",
)

SKIP_REGEX_MARKERS = (
    "unexpected token",
    "unexpected end of input",
    "referenceerror",
    "typeerror",
    "syntaxerror",
    "is not defined",
    "undefined",
    "nan",
    "object object",
    "\\{\\{",
)

REGEX_LITERAL_RE = re.compile(r"/((?:\\.|[^/\\\n])+)/([dgimsuvy]*)")
PATTERNS_ARRAY_RE = re.compile(r"(patterns\s*=\s*\[)(.*?)(\])", re.DOTALL)
STRING_LITERAL_RE = re.compile(r"([\"'])(.*?)(?<!\\)\1", re.DOTALL)
TEXT_INCLUDES_RE = re.compile(r"\b(?P<var>text|t|body)\.includes\((?P<quote>[\"'])(?P<value>[^\"']{2,80})(?P=quote)\)")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def normalize_probe(text: str) -> str:
    return re.sub(r"\\[bBdDsSwW]|\(\?:|\(\?i\)|[\[\]{}^$+*?.()]", " ", text.lower())


def term_present(term: str, probe: str) -> bool:
    escaped = re.escape(term.lower()).replace(r"\ ", r"[\s_\-]*")
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", probe, flags=re.IGNORECASE))


def glossary_alternatives(text: str) -> list[str]:
    probe = normalize_probe(text)
    out: list[str] = []
    for term, alternatives in GLOSSARY.items():
        if term == "turn" and term_present("turn off", probe):
            continue
        if term_present(term, probe):
            out.extend(alternatives)
    unique: list[str] = []
    seen = {part for part in re.split(r"\|", text) if part}
    for alt in out:
        if alt not in seen and alt not in text and alt not in unique:
            unique.append(alt)
    return unique


def append_pattern_alternatives(pattern: str) -> tuple[str, int]:
    alternatives = glossary_alternatives(pattern)
    if not alternatives:
        return pattern, 0
    return pattern + "|" + "|".join(re.escape(alt) for alt in alternatives), len(alternatives)


def is_text_sensitive_expression(expression: str) -> bool:
    return any(marker in expression for marker in TEXT_READ_MARKERS)


def is_source_sensitive_expression(expression: str) -> bool:
    return any(marker in expression for marker in SOURCE_READ_MARKERS)


def should_skip_regex(content: str) -> bool:
    lowered = content.lower()
    return any(marker in lowered for marker in SKIP_REGEX_MARKERS)


def expand_regex_literals(expression: str) -> tuple[str, int]:
    patches = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal patches
        content = match.group(1)
        flags = match.group(2)
        if should_skip_regex(content):
            return match.group(0)
        new_content, added = append_pattern_alternatives(content)
        if added:
            patches += 1
            return f"/{new_content}/{flags}"
        return match.group(0)

    return REGEX_LITERAL_RE.sub(repl, expression), patches


def expand_patterns_arrays(expression: str) -> tuple[str, int]:
    patches = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal patches
        prefix, body, suffix = match.groups()
        additions: list[str] = []
        for literal in STRING_LITERAL_RE.finditer(body):
            value = literal.group(2)
            for alt in glossary_alternatives(value):
                if alt not in additions and alt not in body:
                    additions.append(alt)
        if not additions:
            return match.group(0)
        patches += len(additions)
        sep = "" if body.rstrip().endswith(",") or not body.strip() else ","
        addition_text = ",".join(json.dumps(alt, ensure_ascii=False) for alt in additions)
        return f"{prefix}{body}{sep}{addition_text}{suffix}"

    return PATTERNS_ARRAY_RE.sub(repl, expression), patches


def expand_text_includes(expression: str) -> tuple[str, int]:
    patches = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal patches
        var_name = match.group("var")
        value = match.group("value")
        alternatives = glossary_alternatives(value)
        if not alternatives:
            return match.group(0)
        patches += 1
        clauses = [match.group(0)]
        clauses.extend(f"{var_name}.includes({json.dumps(alt, ensure_ascii=False)})" for alt in alternatives)
        return "(" + "||".join(clauses) + ")"

    return TEXT_INCLUDES_RE.sub(repl, expression), patches


def high_risk_expression(expression: str) -> bool:
    if not is_text_sensitive_expression(expression):
        return False
    term_hits = sum(1 for term in GLOSSARY if term_present(term, normalize_probe(expression)))
    return len(expression) > 2000 or (term_hits >= 4 and ("&&" in expression or "||" in expression))


def bilingualize_step(step: dict[str, Any], stats: Counter[str], risk_examples: list[str], label: str) -> dict[str, Any]:
    action = step.get("action")
    out = copy.deepcopy(step)

    if action in {"assert_text_contains", "assert_text_not_contains"}:
        base = str(out.get("text_pattern") or out.get("text") or "")
        new_pattern, added = append_pattern_alternatives(base)
        if added:
            out["text_pattern"] = new_pattern
            stats["text_assertion_steps_patched"] += 1
            stats["text_assertion_alternatives_added"] += added
        return out

    if action == "click_text":
        base = str(out.get("text_pattern") or out.get("text") or "")
        new_pattern, added = append_pattern_alternatives(base)
        if added:
            out["text_pattern"] = new_pattern
            stats["click_text_steps_patched"] += 1
            stats["click_text_alternatives_added"] += added
        return out

    if action in {"assert_js_value", "eval_js"}:
        expression = str(out.get("expression") or "")
        if not is_text_sensitive_expression(expression):
            return out
        if is_source_sensitive_expression(expression):
            stats["source_text_expressions_skipped"] += 1
            return out
        if high_risk_expression(expression):
            stats["high_risk_text_expressions"] += 1
            if len(risk_examples) < 30:
                risk_examples.append(label)

        expression, regex_patches = expand_regex_literals(expression)
        expression, array_patches = expand_patterns_arrays(expression)
        expression, includes_patches = expand_text_includes(expression)
        if expression != out.get("expression"):
            out["expression"] = expression
            stats["js_steps_patched"] += 1
            stats["js_regex_patches"] += regex_patches
            stats["js_array_alternatives_added"] += array_patches
            stats["js_includes_patches"] += includes_patches
        return out

    return out


def bilingualize_test_cases(test_cases: list[dict[str, Any]], stats: Counter[str], risk_examples: list[str], item_id: str) -> list[dict[str, Any]]:
    out_cases: list[dict[str, Any]] = []
    for tc in test_cases:
        new_tc = copy.deepcopy(tc)
        steps = []
        for idx, step in enumerate(tc.get("steps", []) or []):
            label = f"{item_id}/{tc.get('id')}/step[{idx}]/{step.get('action')}"
            steps.append(bilingualize_step(step, stats, risk_examples, label))
            stats["steps_seen"] += 1
        new_tc["steps"] = steps
        out_cases.append(new_tc)
        stats["test_cases_seen"] += 1
    return out_cases


def sync_file(en_path: Path, zh_path: Path, *, write: bool, stats: Counter[str], risk_examples: list[str]) -> None:
    en_rows = load_jsonl(en_path)
    zh_rows = load_jsonl(zh_path)
    if len(en_rows) != len(zh_rows):
        raise ValueError(f"line count mismatch for {en_path.name}: en={len(en_rows)} zh={len(zh_rows)}")

    new_rows: list[dict[str, Any]] = []
    changed_rows = 0
    for line_no, (en_item, old_zh) in enumerate(zip(en_rows, zh_rows), start=1):
        if en_item.get("id") != old_zh.get("id"):
            raise ValueError(
                f"id mismatch {en_path.name}:{line_no}: en={en_item.get('id')} zh={old_zh.get('id')}"
            )
        new_zh = copy.deepcopy(en_item)
        new_zh["prompt"] = old_zh.get("prompt", "")
        new_zh["language"] = "zh"
        new_zh["test_cases"] = bilingualize_test_cases(
            copy.deepcopy(en_item.get("test_cases") or []),
            stats,
            risk_examples,
            str(en_item.get("id")),
        )
        for field in META_FIELDS:
            if en_item.get(field) != old_zh.get(field):
                stats[f"metadata_synced_{field}"] += 1
        if new_zh != old_zh:
            changed_rows += 1
        new_rows.append(new_zh)
        stats["items_seen"] += 1
    stats[f"changed_rows_{en_path.name}"] = changed_rows
    if write:
        write_jsonl(zh_path, new_rows)


def create_backup(zh_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep backups outside benchmark/zh because the benchmark loader scans that
    # directory recursively, including hidden subdirectories.
    backup_dir = zh_dir.parent / f".backup_zh_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for filename in FILES:
        shutil.copy2(zh_dir / filename, backup_dir / filename)
    return backup_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--en-dir", type=Path, default=Path("benchmark/en"))
    parser.add_argument("--zh-dir", type=Path, default=Path("benchmark/zh"))
    args = parser.parse_args()

    stats: Counter[str] = Counter()
    risk_examples: list[str] = []
    backup_dir: Path | None = None

    if args.write:
        backup_dir = create_backup(args.zh_dir)

    for filename in FILES:
        sync_file(
            args.en_dir / filename,
            args.zh_dir / filename,
            write=args.write,
            stats=stats,
            risk_examples=risk_examples,
        )

    print("sync_zh_benchmark_from_en")
    print(f"mode={'write' if args.write else 'dry-run'}")
    if backup_dir:
        print(f"backup_dir={backup_dir}")
    for key, value in sorted(stats.items()):
        print(f"{key}={value}")
    if risk_examples:
        print("high_risk_examples:")
        for example in risk_examples:
            print(f"  - {example}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
