
# SmartLogix Agent module

import re
import pandas as pd
import requests
from sqlalchemy import create_engine

DB_URL = "postgresql+psycopg2://postgres:data123@localhost:5432/smartlogix_db"

engine = create_engine(DB_URL)

ollama_url = "http://localhost:11434"


def detect_intent(question):
    q = question.lower().strip()

    if re.search(r"\bord-\d+\b", q):
        return "ORDER_QUERY"

    if (
        "capacity" in q
        or "overloaded" in q
        or "weight limit" in q
        or "weight exceeds" in q
    ):
        return "CAPACITY_QUERY"

    if (
        "range" in q
        or "maximum distance" in q
        or "max distance" in q
    ):
        return "RANGE_QUERY"

    if (
        "feasible" in q
        or "can be delivered" in q
        or "deliver by drone" in q
    ):
        return "FEASIBILITY_QUERY"

    return "UNKNOWN"


def retrieve_drone_data(question):

    q = question.lower().strip()

    order_match = re.search(
        r"\bord-\d+\b",
        q,
        re.IGNORECASE
    )

    if order_match:

        order_id = order_match.group(0).upper()

        sql = """
        SELECT
            order_id,
            package_weight,
            vehicle_type,
            capacity_kg,
            distance_km,
            max_range_km,
            weight_exceeds_capacity
        FROM orders_raw
        WHERE UPPER(order_id) = %s
        """

        with engine.connect() as conn:

            return pd.read_sql_query(
                sql,
                conn,
                params=(order_id,)
            )

    if "feasible" in q:

        sql = """
        SELECT
            order_id,
            package_weight,
            vehicle_type,
            capacity_kg,
            distance_km,
            max_range_km,
            weight_exceeds_capacity
        FROM orders_raw
        WHERE LOWER(vehicle_type) = 'drone'
          AND package_weight <= capacity_kg
          AND distance_km <= max_range_km
        """

    elif (
        "capacity" in q
        and (
            "exceed" in q
            or "over" in q
            or "violation" in q
            or "violat" in q
        )
    ):

        sql = """
        SELECT
            order_id,
            package_weight,
            vehicle_type,
            capacity_kg,
            distance_km,
            max_range_km,
            weight_exceeds_capacity
        FROM orders_raw
        WHERE LOWER(vehicle_type) = 'drone'
          AND package_weight > capacity_kg
        """

    elif (
        "range" in q
        or "maximum distance" in q
        or "max distance" in q
    ):

        sql = """
        SELECT
            order_id,
            package_weight,
            vehicle_type,
            capacity_kg,
            distance_km,
            max_range_km,
            weight_exceeds_capacity
        FROM orders_raw
        WHERE LOWER(vehicle_type) = 'drone'
          AND distance_km <= max_range_km
        """

    else:

        return pd.DataFrame()

    with engine.connect() as conn:

        return pd.read_sql_query(
            sql,
            conn
        )


def generate_agent_response(question, retrieved_data):

    prompt = f"""
You are SmartLogix AI, a logistics data assistant.

Answer the user's question using ONLY the verified database result.

User question:
{question}

Verified database result:
{retrieved_data}

Rules:
- Give a short, direct business answer.
- Do not invent information.
- Do not give generic logistics advice.
- For specific orders, use only the retrieved order data.
- If no order is found, clearly say that the order was not found.
"""

    payload = {
        "model": "deepseek-r1:7b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        f"{ollama_url}/api/generate",
        json=payload
    )

    if response.status_code != 200:
        return "Unable to generate an answer from the LLM."

    return response.json().get(
        "response",
        "No response generated."
    )


def smartlogix_agent(question):

    intent = detect_intent(question)

    retrieved_data = retrieve_drone_data(question)

    if intent == "FEASIBILITY_QUERY":

        return (
            f"Based on the verified database, there are "
            f"{len(retrieved_data)} feasible drone routes. "
            f"These routes satisfy both the drone capacity "
            f"and maximum range requirements."
        )

    if intent == "CAPACITY_QUERY":

        return (
            f"Based on the verified database, there are "
            f"{len(retrieved_data)} drone routes with capacity violations. "
            f"These routes have package weights exceeding vehicle capacity."
        )

    if intent == "RANGE_QUERY":

        return (
            f"Based on the verified database, there are "
            f"{len(retrieved_data)} drone routes within their maximum range."
        )

    if retrieved_data.empty:

        return "Order not found in the verified database."

    return generate_agent_response(
        question,
        retrieved_data.to_string(index=False)
    )


print("SmartLogix Agent module created successfully.")
