# test_predict_api_ultra.py
import requests
from tabulate import tabulate
import random
import string
from datetime import datetime
import numpy as np

API_URL = "http://localhost:8000/api/v1/ml/v1/predict_score"

# ========================================================================
# ENHANCED TEST SUITE: 100+ CASES
# ========================================================================
TEST_CASES = []


# Helper: Generate random message with intent
def rand_msg(words, intent_keywords=None):
    base = " ".join(random.choices(words, k=random.randint(3, 12)))
    if intent_keywords and random.random() < 0.6:
        base += " " + random.choice(intent_keywords)
    return base.capitalize()


# -----------------------------------------------------------------------
# 1. COLD LEADS (20 cases) — Low intent, slow, passive
# -----------------------------------------------------------------------
cold_msgs = [
    "just browsing",
    "maybe later",
    "not interested",
    "what is this",
    "send info",
    "hmm",
    "ok",
    "thanks",
    "hello",
    "hi",
    "looking around",
    "not sure",
    "don't know",
    "whatever",
    "k",
    "cool",
    "seen it",
    "bye",
    "no thanks",
    "later",
]
cold_cases = []
for i in range(20):
    msg = random.choice(cold_msgs) if i < 15 else rand_msg(cold_msgs)
    TEST_CASES.append(
        {
            "name": f"Cold {i+1:02d} - {msg.split()[0].capitalize()}...",
            "data": {
                "messages_in_session": random.randint(1, 3),
                "user_msg": msg,
                "conversation_duration_minutes": round(random.uniform(0.5, 3.0), 1),
                "user_response_time_avg_seconds": round(random.uniform(120, 300), 1),
                "user_initiated_conversation": random.choice([False, False, True]),
                "is_returning_customer": False,
                "time_of_day": "business_hours",
            },
        }
    )

# -----------------------------------------------------------------------
# 2. WARM LEADS (25 cases) — Questions, research, moderate engagement
# -----------------------------------------------------------------------
warm_keywords = [
    "demo",
    "pricing",
    "how",
    "compare",
    "features",
    "budget",
    "team",
    "options",
    "tell me",
    "more info",
]
warm_templates = [
    "Can I see a {kw}?",
    "How much is {kw}?",
    "What are your {kw}?",
    "How does it {kw}?",
    "I'm comparing {kw}",
    "Do you have {kw}?",
    "Tell me about {kw}",
    "What’s the {kw}?",
]
for i in range(25):
    kw = random.choice(warm_keywords)
    msg = random.choice(warm_templates).format(kw=kw)
    if i >= 20:
        msg = rand_msg(
            warm_keywords + ["for my team", "for business", "next quarter"], [kw]
        )
    TEST_CASES.append(
        {
            "name": f"Warm {i+1:02d} - {kw.capitalize()} Inquiry",
            "data": {
                "messages_in_session": random.randint(3, 7),
                "user_msg": msg,
                "conversation_duration_minutes": round(random.uniform(4.0, 10.0), 1),
                "user_response_time_avg_seconds": round(random.uniform(60, 120), 1),
                "user_initiated_conversation": True,
                "is_returning_customer": random.choice([False, True]),
                "time_of_day": "business_hours",
            },
        }
    )

# -----------------------------------------------------------------------
# 3. HOT LEADS (20 cases) — Urgency, ready to buy, fast
# -----------------------------------------------------------------------
hot_keywords = [
    "buy",
    "purchase",
    "start",
    "today",
    "ASAP",
    "now",
    "implement",
    "sign",
    "contract",
    "close",
]
hot_templates = [
    "I want to {kw} now",
    "Can we {kw} today?",
    "Ready to {kw}",
    "Let's {kw} ASAP",
    "Need this {kw} immediately",
    "When can we {kw}?",
    "I'm ready to {kw}",
]
for i in range(20):
    kw = random.choice(hot_keywords)
    msg = random.choice(hot_templates).format(kw=kw)
    if i >= 15:
        msg += " " + random.choice(
            ["Let’s do it!", "Send contract", "Credit card ready"]
        )
    TEST_CASES.append(
        {
            "name": f"Hot {i+1:02d} - {kw.capitalize()} Urgency",
            "data": {
                "messages_in_session": random.randint(8, 16),
                "user_msg": msg,
                "conversation_duration_minutes": round(random.uniform(12.0, 30.0), 1),
                "user_response_time_avg_seconds": round(random.uniform(5, 40), 1),
                "user_initiated_conversation": True,
                "is_returning_customer": random.choice([False, True, True]),
                "time_of_day": "business_hours",
            },
        }
    )

# -----------------------------------------------------------------------
# 4. RETURNING CUSTOMERS (15 cases) — Renewal, upgrade, loyalty
# -----------------------------------------------------------------------
return_templates = [
    "Time to renew",
    "Want to upgrade",
    "Need new quote",
    "Add features",
    "Loved it last time",
    "Back for more",
    "Extend contract",
    "Bigger plan",
]
for i in range(15):
    msg = random.choice(return_templates)
    if i >= 10:
        msg += f" — used it {'last year' if i%2==0 else 'last quarter'}"
    TEST_CASES.append(
        {
            "name": f"Return {i+1:02d} - {msg.split()[0]} Action",
            "data": {
                "messages_in_session": random.randint(5, 14),
                "user_msg": msg,
                "conversation_duration_minutes": round(random.uniform(8.0, 25.0), 1),
                "user_response_time_avg_seconds": round(random.uniform(20, 80), 1),
                "user_initiated_conversation": True,
                "is_returning_customer": True,
                "time_of_day": "business_hours",
            },
        }
    )

# -----------------------------------------------------------------------
# 5. EDGE CASES & STRESS TESTS (20 cases)
# -----------------------------------------------------------------------
# Very long message
TEST_CASES.append(
    {
        "name": "Edge 01 - 300+ Char Message",
        "data": {
            "messages_in_session": 6,
            "user_msg": "I've been researching AI-powered CRM solutions for months and your product seems to have the best integration with Slack, Zoom, and Salesforce. Can you confirm GDPR compliance, data residency options, SLA uptime, and whether you support custom webhooks for Zapier? Also interested in API rate limits and whether you have a sandbox environment for testing. Finally, do you offer volume discounts for 500+ seats?".strip(),
            "conversation_duration_minutes": 12.0,
            "user_response_time_avg_seconds": 45.0,
            "user_initiated_conversation": True,
            "is_returning_customer": False,
            "time_of_day": "business_hours",
        },
    }
)

# Emoji spam
TEST_CASES.append(
    {
        "name": "Edge 02 - Emoji Spam",
        "data": {
            "messages_in_session": 10,
            "user_msg": "YES!!! Let's go!!!",
            "conversation_duration_minutes": 20.0,
            "user_response_time_avg_seconds": 8.0,
            "user_initiated_conversation": True,
            "is_returning_customer": False,
            "time_of_day": "business_hours",
        },
    }
)

# Typos & slang
TEST_CASES.append(
    {
        "name": "Edge 03 - Typos & Slang",
        "data": {
            "messages_in_session": 5,
            "user_msg": "helo i wanna by now pls send link asap thx",
            "conversation_duration_minutes": 10.0,
            "user_response_time_avg_seconds": 30.0,
            "user_initiated_conversation": True,
            "is_returning_customer": False,
            "time_of_day": "business_hours",
        },
    }
)

# Numbers & pricing
for i, price in enumerate(["$99", "$500/month", "10k/year", "enterprise"]):
    TEST_CASES.append(
        {
            "name": f"Edge 0{i+4} - Pricing Mention",
            "data": {
                "messages_in_session": 7,
                "user_msg": f"Is {price} the final price? Can we negotiate?",
                "conversation_duration_minutes": 15.0,
                "user_response_time_avg_seconds": 60.0,
                "user_initiated_conversation": True,
                "is_returning_customer": True,
                "time_of_day": "business_hours",
            },
        }
    )

# Ultra-fast (bot-like)
TEST_CASES.append(
    {
        "name": "Edge 08 - Bot Speed",
        "data": {
            "messages_in_session": 20,
            "user_msg": "yes yes yes go go go",
            "conversation_duration_minutes": 10.0,
            "user_response_time_avg_seconds": 0.5,
            "user_initiated_conversation": True,
            "is_returning_customer": False,
            "time_of_day": "business_hours",
        },
    }
)

# Ultra-slow (ghosting)
TEST_CASES.append(
    {
        "name": "Edge 09 - Ghosting",
        "data": {
            "messages_in_session": 1,
            "user_msg": "...",
            "conversation_duration_minutes": 0.1,
            "user_response_time_avg_seconds": 3600.0,
            "user_initiated_conversation": False,
            "is_returning_customer": False,
            "time_of_day": "business_hours",
        },
    }
)

# Random garbage
TEST_CASES.append(
    {
        "name": "Edge 10 - Garbage Input",
        "data": {
            "messages_in_session": 3,
            "user_msg": "".join(random.choices(string.ascii_letters + " ", k=50)),
            "conversation_duration_minutes": 5.0,
            "user_response_time_avg_seconds": 100.0,
            "user_initiated_conversation": True,
            "is_returning_customer": False,
            "time_of_day": "business_hours",
        },
    }
)

# -----------------------------------------------------------------------
# 6. STATISTICAL CORNER CASES (10 cases)
# -----------------------------------------------------------------------
# Near-threshold warm
for i in range(5):
    TEST_CASES.append(
        {
            "name": f"Thresh Warm {i+1}",
            "data": {
                "messages_in_session": 4,
                "user_msg": "Maybe interested in demo",
                "conversation_duration_minutes": 6.0,
                "user_response_time_avg_seconds": 100.0 + i * 10,
                "user_initiated_conversation": True,
                "is_returning_customer": False,
                "time_of_day": "business_hours",
            },
        }
    )

# Near-threshold hot
for i in range(5):
    TEST_CASES.append(
        {
            "name": f"Thresh Hot {i+1}",
            "data": {
                "messages_in_session": 9,
                "user_msg": "Ready to buy if price is right",
                "conversation_duration_minutes": 16.0,
                "user_response_time_avg_seconds": 40.0 - i * 5,
                "user_initiated_conversation": True,
                "is_returning_customer": True,
                "time_of_day": "business_hours",
            },
        }
    )


# ========================================================================
# RUN TESTS
# ========================================================================
def run_tests():
    results = []
    print("=" * 90)
    print(f" ULTRA TEST SUITE: {len(TEST_CASES)} CASES ".center(90, "="))
    print(f" API: {API_URL} ".center(90))
    print("=" * 90 + "\n")

    for i, test in enumerate(TEST_CASES, 1):
        name = test["name"]
        payload = test["data"]

        try:
            resp = requests.post(API_URL, json=payload, timeout=15)

            if resp.status_code == 200:
                data = resp.json()
                score = float(data.get("score", 0))
                cat = data.get("category", "N/A")
                conf = float(data.get("confidence", 0))

                results.append(
                    {
                        "Test Case": name,
                        "Score": f"{score:.4f}",
                        "Category": cat,
                        "Confidence": f"{conf:.4f}",
                        "Status": "Pass" if cat in ["cold", "warm", "hot"] else "Warn",
                    }
                )
                print(f"Pass [{i:3d}/{len(TEST_CASES)}] {name}")
            else:
                err = resp.text[:100]
                results.append(
                    {
                        "Test Case": name,
                        "Score": "ERROR",
                        "Category": f"HTTP {resp.status_code}",
                        "Confidence": "—",
                        "Status": "Fail",
                    }
                )
                print(f"Fail [{i:3d}/{len(TEST_CASES)}] {name} - {err}")

        except Exception as e:
            results.append(
                {
                    "Test Case": name,
                    "Score": "CRASH",
                    "Category": "Exception",
                    "Confidence": "—",
                    "Status": "Fail",
                }
            )
            print(f"Fail [{i:3d}/{len(TEST_CASES)}] {name} - {str(e)[:60]}")

    # === FINAL REPORT ===
    print("\n" + "=" * 90)
    print(" FINAL RESULTS ".center(90, "="))
    print("=" * 90 + "\n")
    print(tabulate(results, headers="keys", tablefmt="grid"))

    passed = [r for r in results if r["Status"] == "Pass"]
    failed = [r for r in results if r["Status"] in ["Fail", "Warn"]]

    print(f"\nPass Successful: {len(passed)}/{len(results)}")
    print(f"Fail Failed: {len(failed)}/{len(results)}")

    if passed:
        cats = {}
        confs = {}
        for r in passed:
            c = r["Category"]
            cats[c] = cats.get(c, 0) + 1
            confs.setdefault(c, []).append(float(r["Confidence"]))

        print("\n" + "-" * 90)
        print(" CATEGORY BREAKDOWN ".center(90))
        for c in sorted(cats):
            cnt = cats[c]
            pct = cnt / len(passed) * 100
            avg_conf = np.mean(confs[c])
            print(f"  {c:8s} → {cnt:3d} ({pct:5.1f}%) | Avg Conf: {avg_conf:.4f}")

    print("\n" + "=" * 90)
    return results


if __name__ == "__main__":
    results = run_tests()
