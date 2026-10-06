"""
gen_query.py
============
1. 为现有 55 条查询加入 sheet_ids_negative（从非正样本中随机抽相同数量）
2. 从 sheets.json 的分组结构生成约 100 条新查询，每条同时含正负样本
3. 输出覆盖原 query.json

用法：
    python3 gen_query.py
"""

import json
import re
import random
from collections import defaultdict

random.seed(42)

SHEETS_PATH = "sheets.json"
QUERY_PATH  = "query.json"
TARGET_NEW  = 100   # 新增查询目标数


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def sample_negatives(pos_ids: list, all_ids: list, n: int) -> list:
    """从 all_ids 中排除 pos_ids，随机抽 n 个作为负样本"""
    pool = [i for i in all_ids if i not in set(pos_ids)]
    n = min(n, len(pool))
    return sorted(random.sample(pool, n))


# ─────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────

with open(SHEETS_PATH, encoding="utf-8") as f:
    sheets = json.load(f)

with open(QUERY_PATH, encoding="utf-8") as f:
    existing_queries = json.load(f)

all_ids = [int(sid) for sid in sheets.keys()]


# ─────────────────────────────────────────────
# Step 1：为现有查询加负样本
# ─────────────────────────────────────────────

for q in existing_queries:
    pos = q["sheet_ids"]
    q["sheet_ids_negative"] = sample_negatives(pos, all_ids, len(pos))

print(f"[Step 1] 为现有 {len(existing_queries)} 条查询添加了负样本")


# ─────────────────────────────────────────────
# Step 2：按类别分组 sheets
# ─────────────────────────────────────────────

fin_company    = defaultdict(list)   # Financial_Statements 公司
synth_company  = defaultdict(list)   # Synthetic_Data__Financial 公司
inv_type       = defaultdict(list)   # Inventory 类型
hr_type        = defaultdict(list)   # HR 类型
sales_quarter  = defaultdict(list)   # Sales 季度
sales_year     = defaultdict(list)   # Sales 年份
stock_year     = defaultdict(list)   # Stock_Market_Data 年份
stock_quarter  = defaultdict(list)   # Stock_Market_Data 季度
proj_type      = defaultdict(list)   # Project 类型
finbiz_all     = []                  # Financial_Business_Data
global_all     = []                  # Global_Stock_Indices
movie_all      = []                  # Movie_Ratings
sentiment_all  = []                  # Text_Sentiment

for sid, s in sheets.items():
    name = s["name"]
    iid  = int(sid)

    if name.startswith("Financial_Statements"):
        m = re.search(r"Finance_(.+?)_\d{4}_", name)
        if m:
            comp = m.group(1).replace("_", " ")
            fin_company[comp].append(iid)

    elif name.startswith("Synthetic_Data__Financial"):
        m = re.search(r"Finance_(.+?)_\d{4}", name)
        if m:
            comp = m.group(1).replace("_", " ")
            synth_company[comp].append(iid)

    elif name.startswith("Inventory_Products"):
        if "Stock_Ledger" in name:
            inv_type["stock ledger"].append(iid)
        elif "Warehouse_Entry" in name:
            inv_type["warehouse entry"].append(iid)
        elif "Warehouse_Exit" in name:
            inv_type["warehouse exit"].append(iid)
        elif "Inventory_Check" in name:
            inv_type["inventory check"].append(iid)

    elif name.startswith("Human_Resources"):
        if "Attendance" in name:
            hr_type["attendance records"].append(iid)
        elif "Employee_Roster" in name:
            hr_type["employee roster"].append(iid)
        elif "Training" in name:
            hr_type["training records"].append(iid)
        elif "Salary" in name:
            hr_type["salary details"].append(iid)

    elif name.startswith("Sales_Data"):
        m_q = re.search(r"Sales_\d{4}_(Q\d)", name)
        m_y = re.search(r"Sales_(\d{4})_Q\d", name)
        if m_q: sales_quarter[m_q.group(1)].append(iid)
        if m_y: sales_year[m_y.group(1)].append(iid)

    elif name.startswith("Stock_Market_Data") or \
         (name.startswith("Synthetic_Data__Stock") and "Stock_Market" in name):
        m_y = re.search(r"Stock_(\d{4})_Q\d", name)
        m_q = re.search(r"Stock_\d{4}_(Q\d)", name)
        if m_y: stock_year[m_y.group(1)].append(iid)
        if m_q: stock_quarter[m_q.group(1)].append(iid)

    elif name.startswith("Project_Management"):
        if "Budget" in name:
            proj_type["project budgets"].append(iid)
        elif "Progress" in name:
            proj_type["project progress"].append(iid)
        elif "Registry" in name:
            proj_type["project registry"].append(iid)

    elif name.startswith("Financial_Business_Data"):
        finbiz_all.append(iid)

    elif name.startswith("Global_Stock_Indices"):
        global_all.append(iid)

    elif name.startswith("Movie_Ratings"):
        movie_all.append(iid)

    elif name.startswith("Text_Sentiment"):
        sentiment_all.append(iid)


# ─────────────────────────────────────────────
# Step 3：查询模板
# ─────────────────────────────────────────────

def make_query(text: str, pos_ids: list) -> dict:
    pos = sorted(pos_ids)
    neg = sample_negatives(pos, all_ids, len(pos))
    return {"query": text, "sheet_ids": pos, "sheet_ids_negative": neg}


new_queries = []

# ── Financial Statements（按公司，多种问法）──
fin_templates = [
    "Retrieve all financial records for {company}",
    "Show balance sheet and income data for {company} across all years",
    "Find all {company} financial statement entries",
    "List all sheets related to {company} financial reporting",
    "Get historical financial data for {company}",
]
for comp, ids in fin_company.items():
    if len(ids) < 2:
        continue
    tmpl = random.choice(fin_templates)
    new_queries.append(make_query(tmpl.format(company=comp), ids))

# ── Synthetic Financial（按公司）──
synth_fin_templates = [
    "Retrieve synthetic financial statement records for {company}",
    "Find augmented balance sheet data for {company}",
    "List all synthetic sheets for {company} financial statements",
]
for comp, ids in synth_company.items():
    if len(ids) < 2:
        continue
    tmpl = random.choice(synth_fin_templates)
    new_queries.append(make_query(tmpl.format(company=comp), ids))

# ── Inventory（按类型）──
inv_templates = [
    "Retrieve all {inv_type} records across all periods",
    "List all {inv_type} sheets from inventory data",
    "Find every {inv_type} entry available in the dataset",
    "Show all {inv_type} tables including monthly breakdowns",
]
for inv_t, ids in inv_type.items():
    tmpl = random.choice(inv_templates)
    new_queries.append(make_query(tmpl.format(inv_type=inv_t), ids))
    # 再加一个跨类型查询（多个 inv_type 组合）
if len(inv_type) >= 2:
    types = list(inv_type.keys())
    combined_ids = inv_type[types[0]] + inv_type[types[1]]
    new_queries.append(make_query(
        f"Find all {types[0]} and {types[1]} inventory records",
        combined_ids
    ))

# ── HR（按类型）──
hr_templates = [
    "Retrieve all {hr_type} sheets",
    "List all {hr_type} records across all years",
    "Find every available {hr_type} table",
    "Show complete {hr_type} data for all periods",
]
for hr_t, ids in hr_type.items():
    tmpl = random.choice(hr_templates)
    new_queries.append(make_query(tmpl.format(hr_type=hr_t), ids))

# HR 跨类型
if len(hr_type) >= 2:
    types = list(hr_type.keys())
    new_queries.append(make_query(
        "Retrieve all employee attendance and salary records",
        hr_type.get("attendance records", []) + hr_type.get("salary details", [])
    ))
    new_queries.append(make_query(
        "List all HR data including rosters and training records",
        hr_type.get("employee roster", []) + hr_type.get("training records", [])
    ))

# ── Sales（按季度）──
sales_q_templates = [
    "Find all {quarter} quarterly sales records across all years",
    "Retrieve every {quarter} sales sheet available",
    "List all {quarter} sales data including product and region details",
    "Show {quarter} sales performance sheets",
]
for q, ids in sales_quarter.items():
    tmpl = random.choice(sales_q_templates)
    new_queries.append(make_query(tmpl.format(quarter=q), ids))

# Sales 按年份
sales_y_templates = [
    "Retrieve all {year} sales records",
    "Find every sales sheet from {year}",
    "List all quarterly sales data from {year}",
]
for yr, ids in sales_year.items():
    tmpl = random.choice(sales_y_templates)
    new_queries.append(make_query(tmpl.format(year=yr), ids))

# ── Stock Market（按年份）──
stock_y_templates = [
    "Find all stock market data for {year}",
    "Retrieve every {year} stock trading sheet",
    "List all {year} equity market records with opening and closing prices",
    "Show all {year} stock sheets including volume and price data",
]
for yr, ids in stock_year.items():
    tmpl = random.choice(stock_y_templates)
    new_queries.append(make_query(tmpl.format(year=yr), ids))

# Stock 按季度
for q, ids in stock_quarter.items():
    new_queries.append(make_query(
        f"Retrieve all {q} stock market records across all years",
        ids
    ))

# ── Project Management（按类型）──
proj_templates = [
    "Find all {proj_type} sheets",
    "Retrieve every available {proj_type} table",
    "List all {proj_type} records across all years",
    "Show complete {proj_type} data",
]
for pt, ids in proj_type.items():
    tmpl = random.choice(proj_templates)
    new_queries.append(make_query(tmpl.format(proj_type=pt), ids))

# Project 跨类型
new_queries.append(make_query(
    "Find all project budget and progress sheets",
    proj_type.get("project budgets", []) + proj_type.get("project progress", [])
))

# ── Financial Business Data ──
if finbiz_all:
    for q_text in [
        "Retrieve all financial business institution records",
        "Find every financial business data sheet with account validity rates",
        "List all institutional financial business records including settlement rates",
        "Show all financial business data sheets",
    ]:
        new_queries.append(make_query(q_text, finbiz_all))

# ── Global Stock Indices ──
if global_all:
    for q_text in [
        "Find all global stock index sheets",
        "Retrieve every international equity index record",
        "List all available global market index data",
    ]:
        new_queries.append(make_query(q_text, global_all))

# ── Movie Ratings ──
if movie_all:
    new_queries.append(make_query(
        "Find all movie rating and review records",
        movie_all
    ))
    new_queries.append(make_query(
        "Retrieve all Chinese movie classification and user rating sheets",
        movie_all
    ))

# ── Text Sentiment ──
if sentiment_all:
    new_queries.append(make_query(
        "Retrieve all text sentiment analysis records including labels",
        sentiment_all
    ))

# ── 跨类别综合查询 ──
cross_category = [
    ("Find all HR and sales performance data",
     hr_type.get("employee roster", [])[:8] + sales_quarter.get("Q1", [])[:8]),
    ("Retrieve inventory stock ledger and warehouse entry records",
     inv_type.get("stock ledger", [])[:10] + inv_type.get("warehouse entry", [])[:10]),
    ("List all project budget and financial statement sheets",
     proj_type.get("project budgets", [])[:8] + list(fin_company.values())[0][:8]),
    ("Find all quarterly stock market and global index data",
     stock_quarter.get("Q1", [])[:8] + global_all[:6]),
    ("Retrieve attendance records and salary details for all years",
     hr_type.get("attendance records", []) + hr_type.get("salary details", [])),
    ("Find all sales and inventory data from 2023 onwards",
     sales_year.get("2023", []) + sales_year.get("2024", []) + inv_type.get("stock ledger", [])[:8]),
    ("List all source and target sheets for stock market data",
     stock_year.get("2023", []) + stock_year.get("2024", [])),
    ("Retrieve all project progress and registry records",
     proj_type.get("project progress", []) + proj_type.get("project registry", [])),
    ("Find all warehouse entry and exit records",
     inv_type.get("warehouse entry", []) + inv_type.get("warehouse exit", [])),
    ("List all inventory check and stock ledger sheets",
     inv_type.get("inventory check", []) + inv_type.get("stock ledger", [])[:12]),
]
for q_text, ids in cross_category:
    if ids:
        new_queries.append(make_query(q_text, ids))


# ─────────────────────────────────────────────
# Step 4：去重 + 合并
# ─────────────────────────────────────────────

# 过滤掉 pos_ids 为空的查询
new_queries = [q for q in new_queries if q["sheet_ids"]]

# 去重（按 query 文本）
existing_texts = {q["query"] for q in existing_queries}
new_queries = [q for q in new_queries if q["query"] not in existing_texts]

# 如果超过目标数，随机截取
if len(new_queries) > TARGET_NEW:
    random.shuffle(new_queries)
    new_queries = new_queries[:TARGET_NEW]

all_queries = existing_queries + new_queries

print(f"[Step 2] 新生成 {len(new_queries)} 条查询")
print(f"[Total]  合并后共 {len(all_queries)} 条查询")

# 统计
pos_counts = [len(q["sheet_ids"]) for q in all_queries]
neg_counts = [len(q["sheet_ids_negative"]) for q in all_queries]
print(f"  正样本平均数: {sum(pos_counts)/len(pos_counts):.1f}")
print(f"  负样本平均数: {sum(neg_counts)/len(neg_counts):.1f}")

# ─────────────────────────────────────────────
# Step 5：写回 query.json
# ─────────────────────────────────────────────

with open(QUERY_PATH, "w", encoding="utf-8") as f:
    json.dump(all_queries, f, ensure_ascii=False, indent=2)

print(f"\n已写入 {QUERY_PATH}")
