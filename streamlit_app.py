import streamlit as st
import os
import pickle
import pandas as pd
import requests
import io
import plotly.express as px
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Optional YOLO/Ultralytics support for drone damage detection
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

load_dotenv()

# Load Transport Mode Prediction Model
TRANSPORT_MODEL_PATH = os.path.join("models", "random_forest.pkl")

with open(TRANSPORT_MODEL_PATH, "rb") as file:
    transport_model = pickle.load(file)

# Load ETA Prediction Model
ETA_MODEL_PATH = os.path.join("models", "eta_model.pkl")

with open(ETA_MODEL_PATH, "rb") as file:
    eta_model = pickle.load(file)

# Load Predictive Maintenance model
MAINTENANCE_MODEL_PATH = os.path.join("models", "maintenance_model.pkl")
maintenance_model = None
if os.path.exists(MAINTENANCE_MODEL_PATH):
    try:
        with open(MAINTENANCE_MODEL_PATH, "rb") as file:
            maintenance_model = pickle.load(file)
    except Exception:
        maintenance_model = None

# Load trained drone damage detection model
DRONE_MODEL_PATH = os.path.join("models", "best.pt")
drone_detection_model = None

if YOLO is not None and os.path.exists(DRONE_MODEL_PATH):
    try:
        drone_detection_model = YOLO(DRONE_MODEL_PATH)
    except Exception:
        drone_detection_model = None

st.set_page_config(
    page_title="SmartLogix AI",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 🔐 SMARTLOGIX AI CUSTOMER / EMPLOYEE LOGIN
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "username" not in st.session_state:
    st.session_state.username = None

SMARTLOGIX_USERS = {
    "customer": {
        "username": "customer",
        "password": "customer123",
        "role": "Customer"
    },
    "employee": {
        "username": "employee",
        "password": "employee123",
        "role": "Employee"
    }
}

if not st.session_state.logged_in:

    st.title("🚚 SmartLogix AI")
    st.subheader("🔐 Login")
    st.caption(
        "Intelligent Multi-Modal Logistics & Autonomous Delivery Platform"
    )

    login_role = st.radio(
        "Select Login Type",
        ["👤 Customer Login", "👨‍💼 Employee Login"],
        horizontal=True
    )

    login_username = st.text_input(
        "Username",
        placeholder="Enter your username"
    )

    login_password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password"
    )

    if st.button("🔐 Login", type="primary", width="stretch"):

        selected_user = (
            SMARTLOGIX_USERS["customer"]
            if login_role == "👤 Customer Login"
            else SMARTLOGIX_USERS["employee"]
        )

        if (
            login_username == selected_user["username"]
            and login_password == selected_user["password"]
        ):
            st.session_state.logged_in = True
            st.session_state.user_role = selected_user["role"]
            st.session_state.username = selected_user["username"]
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

    st.stop()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:data123@localhost:5432/smartlogix_db"
)


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


@st.cache_data(ttl=300)
def load_orders():

    try:

        return pd.read_sql(
            text('SELECT * FROM "orders_raw"'),
            engine
        )

    except Exception as e:

        st.error(
            f"Unable to load orders: {e}"
        )

        return pd.DataFrame()


orders_df = load_orders()

# ============================================================
# 👤 CUSTOMER-SPECIFIC ORDER ACCESS
# ============================================================
# Customers must only see orders belonging to their customer account.
# The customer account is linked to the first valid customer_id found
# in orders_raw for this demo application. Employees continue to see
# the complete orders dataset.

if "customer_id" not in st.session_state:
    st.session_state.customer_id = None

if "customer_id_column" not in st.session_state:
    st.session_state.customer_id_column = None

customer_id_candidates = [
    "customer_id",
    "customerid",
    "customer_code",
    "customer_code_id"
]

if st.session_state.user_role == "Customer":

    detected_customer_column = next(
        (
            column
            for column in customer_id_candidates
            if column in orders_df.columns
        ),
        None
    )

    if detected_customer_column is not None:
        st.session_state.customer_id_column = detected_customer_column

        if st.session_state.customer_id is None and not orders_df.empty:
            valid_customer_ids = (
                orders_df[detected_customer_column]
                .dropna()
                .astype(str)
                .str.strip()
            )
            valid_customer_ids = valid_customer_ids[
                valid_customer_ids.ne("")
            ]

            if not valid_customer_ids.empty:
                st.session_state.customer_id = valid_customer_ids.iloc[0]

        if st.session_state.customer_id is not None:
            customer_orders_df = orders_df[
                orders_df[detected_customer_column]
                .astype(str)
                .str.strip()
                .eq(str(st.session_state.customer_id).strip())
            ].copy()
        else:
            customer_orders_df = orders_df.iloc[0:0].copy()

    else:
        # Never expose all orders to a customer if the customer identifier
        # is not available in the source data.
        customer_orders_df = orders_df.iloc[0:0].copy()

else:
    customer_orders_df = orders_df.copy()

@st.cache_data
def build_product_recommendation_model():

    products = pd.read_csv(
        "cleaned_data/product_catalog_cleaned.csv"
    )

    products = products[
        [
            "product_id",
            "product_name",
            "category",
            "sub_category",
            "tags",
            "avg_rating",
            "stock_qty"
        ]
    ].copy()

    products["feature_text"] = (
        products["category"].astype(str)
        + " "
        + products["sub_category"].astype(str)
        + " "
        + products["tags"].astype(str)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        products["feature_text"]
    )

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )

    return products, similarity_matrix

@st.cache_resource

def build_sentiment_model():

    reviews = pd.read_csv(
        "cleaned_data/customer_reviews_cleaned.csv"
    )

    reviews["review_title"] = reviews["review_title"].fillna("")
    reviews["review_text"] = reviews["review_text"].fillna("")

    reviews["combined_text"] = (
        reviews["review_title"].astype(str)
        + " "
        + reviews["review_text"].astype(str)
    )

    reviews["combined_text"] = (
        reviews["combined_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    sentiment_data = reviews[
        (reviews["combined_text"] != "")
        & (reviews["sentiment_label"].notna())
    ].copy()

    x = sentiment_data["combined_text"]
    y = sentiment_data["sentiment_label"]

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2
    )

    x_tfidf = tfidf.fit_transform(x)

    sentiment_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    sentiment_model.fit(
        x_tfidf,
        y
    )

    return reviews, tfidf, sentiment_model


# ============================================================
# 🧭 ROLE-BASED NAVIGATION
# ============================================================

CUSTOMER_PAGES = [
    "📦 Orders",
    "📍 Delivery Tracking",
    "🤖 AI Assistant",
    "🛍️ Products",
    "⭐ Reviews & Insights",
    "🔔 Notifications"
]

EMPLOYEE_PAGES = [
    "📊 Dashboard",
    "📦 Orders",
    "📍 Delivery Tracking",
    "🚛 Fleet Management",
    "🗺️ Route Optimization",
    "🚁 Drone Management",
    "🔍 Drone Damage Detection",
    "🛠️ Predictive Maintenance",
    "📈 Analytics",
    "🤖 ML Predictions",
    "🤖 AI Assistant",
    "🛍️ Products",
    "⭐ Reviews & Insights",
    "🔔 Notifications"
]

if st.session_state.user_role == "Employee":
    MENU_PAGES = EMPLOYEE_PAGES
else:
    MENU_PAGES = CUSTOMER_PAGES

with st.sidebar:

    st.title("🚚 SmartLogix AI")

    st.caption(
        "Intelligent Logistics Platform"
    )

    page = st.radio(
        "MENU",
        MENU_PAGES,
        label_visibility="visible"
    )

    st.caption("SMARTLOGIX AI")

    st.caption(
        "Autonomous Logistics System"
    )

# ============================================================
# 🔐 CUSTOMER / EMPLOYEE ROLE ACCESS
# ============================================================

if st.session_state.user_role == "Customer" and page not in CUSTOMER_PAGES:
    st.warning("🔒 This page is available only to SmartLogix employees.")
    st.info("Please select an available Customer page from the sidebar.")
    st.stop()

if st.session_state.user_role == "Employee" and page not in EMPLOYEE_PAGES:
    st.warning("🔒 You do not have access to this page.")
    st.stop()

with st.sidebar:
    if st.session_state.user_role == "Customer":
        st.success("👤 Customer Login")
        if st.session_state.customer_id is not None:
            st.caption(f"Customer ID: {st.session_state.customer_id}")
    else:
        st.success("👨‍💼 Employee Login")

    st.caption(f"Username: {st.session_state.username}")

    if st.button("🚪 Logout", width="stretch"):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.session_state.customer_id = None
        st.session_state.customer_id_column = None
        st.rerun()



if page == "📊 Dashboard":

    st.title("SmartLogix Operations Dashboard")

    st.caption(
        "Real-time logistics operations and autonomous delivery overview"
    )

    st.info(
        "🟢 Operations System Online  •  Live order data connected"
)   


    total_orders = len(orders_df)

    delivered = 0

    in_transit = 0

    delayed = 0

    if "order_status" in orders_df.columns:

        status = (
            orders_df["order_status"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        delivered = int(
            status.eq("delivered").sum()
        )

        in_transit = int(
            status.eq("in transit").sum()
        )

        delayed = int(
            pd.to_numeric(
                orders_df["delivery_delay_hours"],
                errors="coerce"
            ).gt(0).sum()
        )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "📦 Total Orders",
            f"{total_orders:,}"
        )


    with col2:

        st.metric(
            "✅ Delivered",
            f"{delivered:,}"
        )


    with col3:

        st.metric(
            "🚚 In Transit",
            f"{in_transit:,}"
        )


    with col4:

        st.metric(
            "⚠️ Delayed",
            f"{delayed:,}"
        )




    left, right = st.columns(2)


    with left:

        st.subheader("Delivery Status")

        if (
            not orders_df.empty
            and "order_status" in orders_df.columns
        ):

            status_data = (
                orders_df["order_status"]
                .astype(str)
                .str.strip()
                .value_counts()
                .reset_index()
            )

            status_data.columns = [
                "Status",
                "Orders"
            ]

            fig = px.bar(
                status_data,
                x="Status",
                y="Orders",
                text="Orders"
            )

            fig.update_layout(
                height=350,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                xaxis_title=None,
                yaxis_title="Orders"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        else:

            st.info(
                "Order status information is unavailable."
            )


    with right:

        st.subheader(
            "Orders by Transport Mode"
        )

        if (
            not orders_df.empty
            and "transport_mode" in orders_df.columns
        ):

            mode_data = (
                orders_df["transport_mode"]
                .astype(str)
                .str.strip()
                .value_counts()
                .reset_index()
            )

            mode_data.columns = [
                "Transport Mode",
                "Orders"
            ]

            fig = px.pie(
                mode_data,
                names="Transport Mode",
                values="Orders",
                hole=0.55
            )

            fig.update_layout(
                height=350,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        else:

            st.info(
                "Transport mode information is unavailable."
            )




    st.subheader("Recent Orders")


    if not orders_df.empty:

        columns_to_show = [
            "order_id",
            "order_status",
            "transport_mode",
            "vehicle_id"
        ]

        available_columns = [
            column
            for column in columns_to_show
            if column in orders_df.columns
        ]

        recent_orders = (
            orders_df[
                available_columns
            ]
            .head(8)
        )

        st.dataframe(
            recent_orders,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No order data available."
        )




    st.subheader(
        "Logistics Operations"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🚛 Vehicle Deliveries",
            f"{len(orders_df):,}"
        )


    with col2:

        if "transport_mode" in orders_df.columns:

            vehicle_orders = orders_df[
                orders_df["transport_mode"]
                .astype(str)
                .str.lower()
                .str.contains(
                    "truck|van",
                    na=False,
                    regex=True
                )
            ]

            vehicle_count = len(
                vehicle_orders
            )

        else:

            vehicle_count = 0


        st.metric(
            "🚚 Ground Transport",
            f"{vehicle_count:,}"
        )


    with col3:

        if "transport_mode" in orders_df.columns:

            drone_orders = orders_df[
                orders_df["transport_mode"]
                .astype(str)
                .str.lower()
                .str.contains(
                    "drone",
                    na=False
                )
            ]

            drone_count = len(
                drone_orders
            )

        else:

            drone_count = 0


        st.metric(
            "🚁 Drone Deliveries",
            f"{drone_count:,}"
        )


    with col4:

        if "vehicle_id" in orders_df.columns:

            assigned = int(
                orders_df["vehicle_id"]
                .notna()
                .sum()
            )

        else:

            assigned = 0


        st.metric(
            "📍 Assigned Vehicles",
            f"{assigned:,}"
        )

## Step 1: Build the Orders page 📦

if page == "📦 Orders":

    # Customer view is always restricted to customer_orders_df.
    # Employees continue to use the complete orders_df below.
    visible_orders_df = (
        customer_orders_df
        if st.session_state.user_role == "Customer"
        else orders_df
    )

    st.title("📦 Order Management")

    st.caption(
        "🔎 Search, inspect and manage logistics orders"
    )


    if st.session_state.user_role == "Customer":
        st.caption(
            f"Showing only orders for Customer ID: {st.session_state.customer_id}"
        )


    if visible_orders_df.empty:

        st.warning("⚠️ No order data available.")

    else:

        st.subheader("🔍 Select Your Order")

        if st.session_state.user_role == "Customer":

            customer_order_ids = (
                visible_orders_df["order_id"]
                .dropna()
                .astype(str)
                .str.strip()
                .drop_duplicates()
                .tolist()
            )

            search_order = st.selectbox(
                "🆔 Select your Order ID",
                ["Select your Order ID"] + customer_order_ids,
                index=0,
                key="customer_order_selector"
            )

            if search_order == "Select your Order ID":
                search_order = ""

        else:

            search_order = st.text_input(
                "🆔 Enter Order ID",
                placeholder="Example: ORD-005379",
                key="employee_order_search"
            )

        if search_order:

            search_order = search_order.strip()

            result = visible_orders_df[
                visible_orders_df["order_id"]
                .astype(str)
                .str.upper()
                .eq(search_order.upper())
            ]

            if result.empty:

                if st.session_state.user_role == "Customer":
                    st.error(
                        "❌ This Order ID was not found in your customer account."
                    )
                else:
                    st.error(
                        f"❌ Order {search_order} was not found."
                    )

            else:

                order = result.iloc[0]

                st.success(
                    f"✅ Order {search_order} found successfully"
                )


                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "📦 Order ID",
                        str(
                            order.get(
                                "order_id",
                                "N/A"
                            )
                        )
                    )

                with col2:

                    st.metric(
                        "📍 Status",
                        str(
                            order.get(
                                "order_status",
                                "N/A"
                            )
                        ).title()
                    )

                with col3:

                    st.metric(
                        "🚚 Transport",
                        str(
                            order.get(
                                "transport_mode",
                                "N/A"
                            )
                        ).title()
                    )

                with col4:

                    st.metric(
                        "🚛 Vehicle",
                        str(
                            order.get(
                                "vehicle_id",
                                "N/A"
                            )
                        )
                    )


                st.subheader("📋 Order Details")

                col1, col2 = st.columns(2)

                with col1:

                    st.write("**🆔 Order ID**")
                    st.write(
                        order.get(
                            "order_id",
                            "N/A"
                        )
                    )

                    st.write("**📍 Order Status**")
                    st.write(
                        str(
                            order.get(
                                "order_status",
                                "N/A"
                            )
                        ).title()
                    )

                    st.write("**🚚 Mode of Transport**")
                    st.write(
                        str(
                            order.get(
                                "transport_mode",
                                "N/A"
                            )
                        ).title()
                    )

                    st.write("**🚛 Vehicle ID**")
                    st.write(
                        order.get(
                            "vehicle_id",
                            "N/A"
                        )
                    )

                with col2:

                    st.write("**🚐 Vehicle Type**")
                    st.write(
                        order.get(
                            "vehicle_type",
                            "N/A"
                        )
                    )

                    st.write("**🔢 Vehicle Count**")
                    st.write(
                        order.get(
                            "vehicle_count",
                            "N/A"
                        )
                    )

                    st.write("**⚡ Average Vehicle Speed**")
                    st.write(
                        order.get(
                            "vehicle_avg_speed_kmph",
                            "N/A"
                        )
                    )

                    st.write("**📊 Capacity Utilization**")
                    st.write(
                        order.get(
                            "vehicle_capacity_utilization_pct",
                            "N/A"
                        )
                    )


                st.subheader("🗂️ Complete Order Record")

                st.dataframe(
                    result.T.rename(
                        columns={
                            result.index[0]: "Value"
                        }
                    ),
                    width="stretch"
                )

        else:

            if st.session_state.user_role == "Customer":

                st.info(
                    "🔐 Your orders are hidden for privacy. Enter your Order ID above to view that order."
                )

            else:

                st.subheader("📦 All Orders")

                display_columns = [
                    "order_id",
                    "order_status",
                    "transport_mode",
                    "vehicle_id",
                    "vehicle_type",
                    "vehicle_count"
                ]

                available_columns = [
                    column
                    for column in display_columns
                    if column in visible_orders_df.columns
                ]

                st.dataframe(
                    visible_orders_df[available_columns],
                    width="stretch",
                    hide_index=True,
                    height=500
                )

## Delivery Tracking 📍

if page == "📍 Delivery Tracking":

    # Customer tracking is restricted to the logged-in customer's orders.
    visible_tracking_df = (
        customer_orders_df
        if st.session_state.user_role == "Customer"
        else orders_df
    )

    st.title("📍 Delivery Tracking")

    st.caption(
        "Track shipment status, transport mode and assigned vehicle"
    )


    if st.session_state.user_role == "Customer":
        st.caption(
            f"Showing only deliveries for Customer ID: {st.session_state.customer_id}"
        )


    if visible_tracking_df.empty:

        st.warning("⚠️ No order data available.")

    else:

        st.subheader("🔍 Select Your Delivery")

        if st.session_state.user_role == "Customer":

            customer_delivery_ids = (
                visible_tracking_df["order_id"]
                .dropna()
                .astype(str)
                .str.strip()
                .drop_duplicates()
                .tolist()
            )

            tracking_order = st.selectbox(
                "🆔 Select your Order ID",
                ["Select your Order ID"] + customer_delivery_ids,
                index=0,
                key="customer_delivery_selector"
            )

            if tracking_order == "Select your Order ID":
                tracking_order = ""

        else:

            tracking_order = st.text_input(
                "🆔 Enter Order ID",
                placeholder="Example: ORD-005379",
                key="delivery_tracking_order"
            )

        if tracking_order:

            tracking_order = tracking_order.strip()

            result = visible_tracking_df[
                visible_tracking_df["order_id"]
                .astype(str)
                .str.upper()
                .eq(tracking_order.upper())
            ]

            if result.empty:

                if st.session_state.user_role == "Customer":
                    st.error(
                        "❌ This Order ID was not found in your customer account."
                    )
                else:
                    st.error(
                        f"❌ Order {tracking_order} was not found."
                    )

            else:

                order = result.iloc[0]

                status_value = str(
                    order.get(
                        "order_status",
                        "N/A"
                    )
                ).title()

                transport_value = str(
                    order.get(
                        "transport_mode",
                        "N/A"
                    )
                ).title()

                vehicle_value = str(
                    order.get(
                        "vehicle_id",
                        "N/A"
                    )
                )

                st.success(
                    f"✅ Tracking information found for {tracking_order}"
                )


                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "📦 Order",
                        str(
                            order.get(
                                "order_id",
                                "N/A"
                            )
                        )
                    )

                with col2:

                    st.metric(
                        "📍 Delivery Status",
                        status_value
                    )

                with col3:

                    st.metric(
                        "🚚 Transport Mode",
                        transport_value
                    )

                with col4:

                    st.metric(
                        "🚛 Vehicle ID",
                        vehicle_value
                    )


                st.subheader("📋 Delivery Information")

                col1, col2 = st.columns(2)

                with col1:

                    st.write("**📦 Order ID**")
                    st.write(
                        order.get(
                            "order_id",
                            "N/A"
                        )
                    )

                    st.write("**📍 Current Status**")
                    st.write(status_value)

                    st.write("**🚚 Mode of Transport**")
                    st.write(transport_value)

                    st.write("**🚛 Vehicle ID**")
                    st.write(vehicle_value)

                with col2:

                    st.write("**🚐 Vehicle Type**")
                    st.write(
                        order.get(
                            "vehicle_type",
                            "N/A"
                        )
                    )

                    st.write("**🔢 Vehicle Count**")
                    st.write(
                        order.get(
                            "vehicle_count",
                            "N/A"
                        )
                    )

                    st.write("**⚡ Average Speed**")
                    st.write(
                        order.get(
                            "vehicle_avg_speed_kmph",
                            "N/A"
                        )
                    )

                    st.write("**📊 Capacity Utilization**")
                    st.write(
                        order.get(
                            "vehicle_capacity_utilization_pct",
                            "N/A"
                        )
                    )


                # ========================================================
                # 🗺️ CUSTOMER / EMPLOYEE DELIVERY ROUTE MAP
                # ========================================================
                # This map shows the road route for the selected delivery.
                # It is a route map, not live GPS tracking unless the source
                # dataset contains real-time vehicle coordinates.
                st.subheader("🗺️ Delivery Route Map")

                route_origin_column = next(
                    (
                        column
                        for column in [
                            "origin",
                            "source",
                            "pickup_location",
                            "origin_city"
                        ]
                        if column in order.index
                    ),
                    None
                )

                route_destination_column = next(
                    (
                        column
                        for column in [
                            "destination",
                            "delivery_location",
                            "destination_city"
                        ]
                        if column in order.index
                    ),
                    None
                )

                # Common Indian city coordinates.
                DELIVERY_MAP_CITY_COORDINATES = {
                    "Ahmedabad": (23.0225, 72.5714),
                    "Bengaluru": (12.9716, 77.5946),
                    "Bangalore": (12.9716, 77.5946),
                    "Bhopal": (23.2599, 77.4126),
                    "Bhubaneswar": (20.2961, 85.8245),
                    "Chandigarh": (30.7333, 76.7794),
                    "Chennai": (13.0827, 80.2707),
                    "Coimbatore": (11.0168, 76.9558),
                    "Delhi": (28.6139, 77.2090),
                    "Gurugram": (28.4595, 77.0266),
                    "Gurgaon": (28.4595, 77.0266),
                    "Hyderabad": (17.3850, 78.4867),
                    "Jaipur": (26.9124, 75.7873),
                    "Kanpur": (26.4499, 80.3319),
                    "Kochi": (9.9312, 76.2673),
                    "Kolkata": (22.5726, 88.3639),
                    "Lucknow": (26.8467, 80.9462),
                    "Madurai": (9.9252, 78.1198),
                    "Mumbai": (19.0760, 72.8777),
                    "Mysuru": (12.2958, 76.6394),
                    "Mysore": (12.2958, 76.6394),
                    "Nagpur": (21.1458, 79.0882),
                    "Nashik": (19.9975, 73.7898),
                    "Noida": (28.5355, 77.3910),
                    "Patna": (25.5941, 85.1376),
                    "Pune": (18.5204, 73.8567),
                    "Salem": (11.6643, 78.1460),
                    "Surat": (21.1702, 72.8311),
                    "Thiruvananthapuram": (8.5241, 76.9366),
                    "Trivandrum": (8.5241, 76.9366),
                    "Tiruchirappalli": (10.7905, 78.7047),
                    "Trichy": (10.7905, 78.7047),
                    "Vadodara": (22.3072, 73.1812),
                    "Visakhapatnam": (17.6868, 83.2185)
                }

                def delivery_map_coordinates(value):
                    value_text = str(value).strip()

                    if not value_text:
                        return None

                    # Exact city match, case-insensitive.
                    for city_name, coordinates in DELIVERY_MAP_CITY_COORDINATES.items():
                        if value_text.lower() == city_name.lower():
                            return coordinates

                    # Handle values such as "Chennai, Tamil Nadu".
                    city_part = value_text.split(",")[0].strip()
                    for city_name, coordinates in DELIVERY_MAP_CITY_COORDINATES.items():
                        if city_part.lower() == city_name.lower():
                            return coordinates

                    return None

                if route_origin_column and route_destination_column:

                    delivery_origin = order.get(route_origin_column)
                    delivery_destination = order.get(route_destination_column)

                    origin_coordinates = delivery_map_coordinates(delivery_origin)

                    # Use the actual destination latitude/longitude stored in the ORDERS dataset.
                    # A small number of source rows can contain swapped/invalid coordinates,
                    # so validate them before plotting.
                    destination_lat = pd.to_numeric(
                        order.get("destination_lat"), errors="coerce"
                    )
                    destination_lon = pd.to_numeric(
                        order.get("destination_lon"), errors="coerce"
                    )

                    destination_coordinates_valid = (
                        pd.notna(destination_lat)
                        and pd.notna(destination_lon)
                        and 5 <= float(destination_lat) <= 37
                        and 68 <= float(destination_lon) <= 98
                    )

                    if destination_coordinates_valid:
                        destination_coordinates = (
                            float(destination_lat),
                            float(destination_lon)
                        )
                        destination_from_dataset = True
                    else:
                        destination_coordinates = delivery_map_coordinates(
                            delivery_destination
                        )
                        destination_from_dataset = False

                    if origin_coordinates and destination_coordinates:

                        origin_lat, origin_lon = origin_coordinates
                        destination_lat, destination_lon = destination_coordinates

                        if destination_from_dataset:
                            st.caption(
                                "📍 Destination uses latitude/longitude from the ORDERS dataset."
                            )
                        else:
                            st.warning(
                                "⚠️ The selected order has an invalid destination coordinate, "
                                "so the destination city coordinate is being used as a fallback."
                            )

                        route_distance_km = None
                        route_duration_hours = None
                        route_geometry = None

                        try:
                            osrm_url = (
                                "https://router.project-osrm.org/route/v1/driving/"
                                f"{origin_lon},{origin_lat};"
                                f"{destination_lon},{destination_lat}"
                                "?overview=full&geometries=geojson"
                            )

                            osrm_response = requests.get(
                                osrm_url,
                                timeout=10
                            )
                            osrm_response.raise_for_status()
                            osrm_data = osrm_response.json()

                            if osrm_data.get("routes"):
                                selected_route = osrm_data["routes"][0]
                                route_distance_km = (
                                    float(selected_route.get("distance", 0)) / 1000
                                )
                                route_duration_hours = (
                                    float(selected_route.get("duration", 0)) / 3600
                                )
                                route_geometry = selected_route.get("geometry")

                        except Exception:
                            route_distance_km = None
                            route_duration_hours = None
                            route_geometry = None

                        if route_distance_km is None:
                            distance_value = order.get("distance_km")
                            try:
                                route_distance_km = float(distance_value)
                            except (TypeError, ValueError):
                                route_distance_km = None

                        map_col1, map_col2, map_col3 = st.columns(3)

                        with map_col1:
                            st.metric(
                                "📍 From",
                                str(delivery_origin)
                            )

                        with map_col2:
                            st.metric(
                                "🏁 To",
                                str(delivery_destination)
                            )

                        with map_col3:
                            if route_distance_km is not None:
                                st.metric(
                                    "📏 Route Distance",
                                    f"{route_distance_km:,.1f} km"
                                )
                            else:
                                st.metric(
                                    "📏 Route Distance",
                                    "N/A"
                                )

                        if route_duration_hours is not None:
                            st.caption(
                                f"Estimated road travel time: "
                                f"{route_duration_hours:.1f} hours"
                            )

                        try:
                            import pydeck as pdk

                            point_data = pd.DataFrame(
                                [
                                    {
                                        "name": "Origin",
                                        "longitude": origin_lon,
                                        "latitude": origin_lat,
                                        "type": "Origin"
                                    },
                                    {
                                        "name": "Destination",
                                        "longitude": destination_lon,
                                        "latitude": destination_lat,
                                        "type": "Destination"
                                    }
                                ]
                            )

                            layers = [
                                pdk.Layer(
                                    "ScatterplotLayer",
                                    data=point_data,
                                    get_position="[longitude, latitude]",
                                    get_radius=9000,
                                    get_fill_color="[30, 144, 255, 220]",
                                    pickable=True
                                )
                            ]

                            if route_geometry and route_geometry.get("coordinates"):
                                route_coordinates = [
                                    [coordinate[0], coordinate[1]]
                                    for coordinate in route_geometry["coordinates"]
                                ]

                                route_line_data = pd.DataFrame(
                                    [{"route": route_coordinates}]
                                )

                                layers.insert(
                                    0,
                                    pdk.Layer(
                                        "PathLayer",
                                        data=route_line_data,
                                        get_path="route",
                                        get_width=7,
                                        get_color="[0, 102, 255, 230]",
                                        width_min_pixels=4,
                                        pickable=True
                                    )
                                )
                            else:
                                straight_route_data = pd.DataFrame(
                                    [
                                        {
                                            "route": [
                                                [origin_lon, origin_lat],
                                                [destination_lon, destination_lat]
                                            ]
                                        }
                                    ]
                                )

                                layers.insert(
                                    0,
                                    pdk.Layer(
                                        "PathLayer",
                                        data=straight_route_data,
                                        get_path="route",
                                        get_width=7,
                                        get_color="[0, 102, 255, 230]",
                                        width_min_pixels=4
                                    )
                                )

                            midpoint_lat = (origin_lat + destination_lat) / 2
                            midpoint_lon = (origin_lon + destination_lon) / 2

                            lat_difference = abs(destination_lat - origin_lat)
                            lon_difference = abs(destination_lon - origin_lon)
                            max_difference = max(lat_difference, lon_difference)

                            if max_difference > 20:
                                zoom_level = 3.5
                            elif max_difference > 10:
                                zoom_level = 4.5
                            elif max_difference > 5:
                                zoom_level = 5.5
                            else:
                                zoom_level = 7

                            delivery_view_state = pdk.ViewState(
                                latitude=midpoint_lat,
                                longitude=midpoint_lon,
                                zoom=zoom_level,
                                pitch=0
                            )

                            st.pydeck_chart(
                                pdk.Deck(
                                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                                    initial_view_state=delivery_view_state,
                                    layers=layers,
                                    tooltip={
                                        "html": "<b>{name}</b>",
                                        "style": {
                                            "backgroundColor": "white",
                                            "color": "black"
                                        }
                                    }
                                ),
                                width="stretch",
                                height=500
                            )

                        except ImportError:
                            map_dataframe = pd.DataFrame(
                                {
                                    "lat": [origin_lat, destination_lat],
                                    "lon": [origin_lon, destination_lon]
                                }
                            )

                            st.map(
                                map_dataframe,
                                latitude="lat",
                                longitude="lon",
                                zoom=None,
                                use_container_width=True
                            )

                        if route_geometry:
                            st.success(
                                "🛣️ Road route displayed between the pickup location and delivery destination."
                            )
                        else:
                            st.info(
                                "ℹ️ Road routing service was unavailable, so the map shows a direct route between the two locations."
                            )

                    else:
                        st.info(
                            "🗺️ A delivery map will appear when the selected order's origin and destination cities have supported map coordinates."
                        )

                else:
                    st.info(
                        "🗺️ Origin and destination information is not available for this order, so a delivery map cannot be displayed."
                    )

                st.subheader("🗂️ Tracking Record")

                tracking_columns = [
                    "order_id",
                    "order_status",
                    "transport_mode",
                    "vehicle_id",
                    "vehicle_type",
                    "vehicle_count",
                    "vehicle_avg_speed_kmph",
                    "vehicle_capacity_utilization_pct"
                ]

                available_columns = [
                    column
                    for column in tracking_columns
                    if column in result.columns
                ]

                st.dataframe(
                    result[available_columns],
                    width="stretch",
                    hide_index=True
                )

        else:

            if st.session_state.user_role == "Customer":

                st.info(
                    "🔐 Your deliveries are hidden for privacy. Enter your Order ID above to track that delivery."
                )

            else:

                st.subheader("📦 Active Shipment Overview")

                status_columns = [
                    "order_id",
                    "order_status",
                    "transport_mode",
                    "vehicle_id"
                ]

                available_columns = [
                    column
                    for column in status_columns
                    if column in visible_tracking_df.columns
                ]

                st.dataframe(
                    visible_tracking_df[available_columns].head(15),
                    width="stretch",
                    hide_index=True,
                    height=500
                )

## 🚛 Fleet Management

if page == "🚛 Fleet Management":

    st.title("🚛 Fleet Management")

    st.caption(
        "Monitor vehicles, transportation modes and fleet utilization"
    )


    st.subheader("📊 Fleet Overview")

    total_vehicles = 0
    ground_vehicles = 0
    drone_vehicles = 0

    if not orders_df.empty:

        if "vehicle_id" in orders_df.columns:
            total_vehicles = orders_df["vehicle_id"].nunique()

        if "transport_mode" in orders_df.columns:

            transport_values = (
                orders_df["transport_mode"]
                .astype(str)
                .str.lower()
            )

            ground_vehicles = orders_df[
                transport_values != "drone"
            ]["vehicle_id"].nunique()

            drone_vehicles = orders_df[
                transport_values == "drone"
            ]["vehicle_id"].nunique()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🚛 Active Vehicles",
            f"{total_vehicles:,}"
        )

    with col2:

        st.metric(
            "🚚 Ground Vehicles",
            f"{ground_vehicles:,}"
        )

    with col3:

        st.metric(
            "🚁 Drone Vehicles",
            f"{drone_vehicles:,}"
        )


    st.subheader("🚚 Transportation Distribution")

    if not orders_df.empty and "transport_mode" in orders_df.columns:

        transport_summary = (
            orders_df["transport_mode"]
            .astype(str)
            .str.title()
            .value_counts()
            .reset_index()
        )

        transport_summary.columns = [
            "Transport Mode",
            "Orders"
        ]

        fig = px.bar(
            transport_summary,
            x="Transport Mode",
            y="Orders",
            title="Orders by Transportation Mode"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.info(
            "ℹ️ Transportation data is not available."
        )


    st.subheader("🚛 Fleet Vehicle Records")

    if orders_df.empty:

        st.warning(
            "⚠️ No fleet data available."
        )

    else:

        fleet_columns = [
            "vehicle_id",
            "vehicle_type",
            "transport_mode",
            "vehicle_count",
            "vehicle_avg_speed_kmph",
            "vehicle_capacity_utilization_pct"
        ]

        available_columns = [
            column
            for column in fleet_columns
            if column in orders_df.columns
        ]

        fleet_table = (
            orders_df[available_columns]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        st.dataframe(
            fleet_table,
            width="stretch",
            hide_index=True,
            height=500
        )

## 🗺️ Route Optimization

if page == "🗺️ Route Optimization":

    if st.session_state.user_role == "Customer":
        st.warning("🔒 Route Optimization is available only to SmartLogix employees.")
        st.stop()


    visible_route_orders_df = (
        customer_orders_df
        if st.session_state.user_role == "Customer"
        else orders_df
    )

    st.title("🗺️ Route Optimization")

    st.caption(
        "Analyze transportation routes, distances and delivery assignments"
    )


    if visible_route_orders_df.empty:

        st.warning("⚠️ No order data available.")

    else:

        st.subheader("📊 Route Overview")

        distance_columns = [
            "route_distance_km",
            "distance_km",
            "delivery_distance_km",
            "total_distance_km"
        ]

        distance_column = next(
            (
                column
                for column in distance_columns
                if column in visible_route_orders_df.columns
            ),
            None
        )

        total_orders = len(visible_route_orders_df)

        transport_count = (
            visible_route_orders_df["transport_mode"].nunique()
            if "transport_mode" in visible_route_orders_df.columns
            else 0
        )

        unique_routes = total_orders

        if distance_column:

            distance_data = pd.to_numeric(
                visible_route_orders_df[distance_column],
                errors="coerce"
            )

            average_distance = distance_data.mean()
            total_distance = distance_data.sum()

        else:

            average_distance = None
            total_distance = None

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "📦 Orders",
                f"{total_orders:,}"
            )

        with col2:

            st.metric(
                "🛣️ Routes",
                f"{unique_routes:,}"
            )

        with col3:

            st.metric(
                "🚚 Transport Modes",
                f"{transport_count:,}"
            )

        with col4:

            if average_distance is not None:

                st.metric(
                    "📏 Avg Distance",
                    f"{average_distance:,.1f} km"
                )

            else:

                st.metric(
                    "📏 Avg Distance",
                    "N/A"
                )


        st.subheader("📏 Route Distance Analysis")

        if distance_column:

            route_distance_data = pd.DataFrame({
                "Distance (km)": pd.to_numeric(
                    visible_route_orders_df[distance_column],
                    errors="coerce"
                )
            }).dropna()

            if not route_distance_data.empty:

                fig = px.histogram(
                    route_distance_data,
                    x="Distance (km)",
                    title="Distribution of Delivery Route Distances",
                    nbins=30
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "📏 Total Distance",
                        f"{total_distance:,.1f} km"
                    )

                with col2:

                    st.metric(
                        "⬇️ Shortest Route",
                        f"{route_distance_data['Distance (km)'].min():,.1f} km"
                    )

                with col3:

                    st.metric(
                        "⬆️ Longest Route",
                        f"{route_distance_data['Distance (km)'].max():,.1f} km"
                    )

            else:

                st.info(
                    "ℹ️ Route distance values are not available."
                )

        else:

            st.info(
                "ℹ️ No route distance column is available in the current dataset."
            )


        # ============================================================
        # 🗺️ INTERACTIVE ROUTE MAP - ADDED WITHOUT CHANGING EXISTING
        # ============================================================

        st.subheader("🗺️ Interactive Route Map")
        st.caption(
            "Select an origin and destination from the available route data to visualize the delivery route."
        )

        route_origin_column = next(
            (
                column
                for column in [
                    "origin",
                    "source",
                    "pickup_location",
                    "origin_city"
                ]
                if column in visible_route_orders_df.columns
            ),
            None
        )

        route_destination_column = next(
            (
                column
                for column in [
                    "destination",
                    "delivery_location",
                    "destination_city"
                ]
                if column in visible_route_orders_df.columns
            ),
            None
        )

        # Common Indian city coordinates used when the dataset stores city names.
        # Unknown cities are handled by the fallback message below.
        SMARTLOGIX_CITY_COORDINATES = {
            "Ahmedabad": (23.0225, 72.5714),
            "Bengaluru": (12.9716, 77.5946),
            "Bangalore": (12.9716, 77.5946),
            "Bhopal": (23.2599, 77.4126),
            "Bhubaneswar": (20.2961, 85.8245),
            "Chandigarh": (30.7333, 76.7794),
            "Chennai": (13.0827, 80.2707),
            "Coimbatore": (11.0168, 76.9558),
            "Delhi": (28.6139, 77.2090),
            "Gurugram": (28.4595, 77.0266),
            "Gurgaon": (28.4595, 77.0266),
            "Hyderabad": (17.3850, 78.4867),
            "Jaipur": (26.9124, 75.7873),
            "Kanpur": (26.4499, 80.3319),
            "Kochi": (9.9312, 76.2673),
            "Kolkata": (22.5726, 88.3639),
            "Lucknow": (26.8467, 80.9462),
            "Madurai": (9.9252, 78.1198),
            "Mumbai": (19.0760, 72.8777),
            "Mysuru": (12.2958, 76.6394),
            "Mysore": (12.2958, 76.6394),
            "Nagpur": (21.1458, 79.0882),
            "Nashik": (19.9975, 73.7898),
            "Noida": (28.5355, 77.3910),
            "Patna": (25.5941, 85.1376),
            "Pune": (18.5204, 73.8567),
            "Salem": (11.6643, 78.1460),
            "Surat": (21.1702, 72.8311),
            "Thiruvananthapuram": (8.5241, 76.9366),
            "Trivandrum": (8.5241, 76.9366),
            "Tiruchirappalli": (10.7905, 78.7047),
            "Trichy": (10.7905, 78.7047),
            "Vadodara": (22.3072, 73.1812),
            "Visakhapatnam": (17.6868, 83.2185)
        }

        def smartlogix_normalize_city_name(value):
            text_value = str(value).strip()
            if not text_value:
                return text_value

            for city_name in SMARTLOGIX_CITY_COORDINATES:
                if text_value.lower() == city_name.lower():
                    return city_name

            return text_value

        def smartlogix_get_city_coordinates(value):
            city_name = smartlogix_normalize_city_name(value)

            if city_name in SMARTLOGIX_CITY_COORDINATES:
                return SMARTLOGIX_CITY_COORDINATES[city_name]

            # Handle strings such as "Chennai, Tamil Nadu".
            city_part = city_name.split(",")[0].strip()

            for known_city, coordinates in SMARTLOGIX_CITY_COORDINATES.items():
                if city_part.lower() == known_city.lower():
                    return coordinates

            return None

        if route_origin_column and route_destination_column:

            route_map_data = visible_route_orders_df[
                [route_origin_column, route_destination_column]
            ].dropna().copy()

            route_map_data[route_origin_column] = (
                route_map_data[route_origin_column].astype(str).str.strip()
            )

            route_map_data[route_destination_column] = (
                route_map_data[route_destination_column].astype(str).str.strip()
            )

            route_map_data = route_map_data[
                (route_map_data[route_origin_column] != "")
                & (route_map_data[route_destination_column] != "")
            ]

            if not route_map_data.empty:

                origin_options = sorted(
                    route_map_data[route_origin_column].unique().tolist()
                )

                selected_map_origin = st.selectbox(
                    "📍 Origin",
                    origin_options,
                    key="smartlogix_map_origin"
                )

                destination_options = sorted(
                    route_map_data[
                        route_map_data[route_map_data.columns[0]] == selected_map_origin
                    ][route_destination_column].unique().tolist()
                )

                if not destination_options:
                    destination_options = sorted(
                        route_map_data[route_destination_column].unique().tolist()
                    )

                selected_map_destination = st.selectbox(
                    "🏁 Destination",
                    destination_options,
                    key="smartlogix_map_destination"
                )

                # Origin coordinates are not present in the ORDERS table.
                # Therefore the origin city/hub is resolved to its city coordinate.
                origin_coordinates = smartlogix_get_city_coordinates(
                    selected_map_origin
                )

                # Destination coordinates ARE stored in ORDERS. Use the actual dataset
                # coordinates for the selected origin/destination pair.
                matching_routes = visible_route_orders_df[
                    (
                        visible_route_orders_df[route_origin_column]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        == str(selected_map_origin).strip().lower()
                    )
                    & (
                        visible_route_orders_df[route_destination_column]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        == str(selected_map_destination).strip().lower()
                    )
                ].copy()

                destination_coordinates = None
                destination_from_dataset = False

                if not matching_routes.empty and {"destination_lat", "destination_lon"}.issubset(
                    matching_routes.columns
                ):
                    destination_lat_values = pd.to_numeric(
                        matching_routes["destination_lat"], errors="coerce"
                    )
                    destination_lon_values = pd.to_numeric(
                        matching_routes["destination_lon"], errors="coerce"
                    )

                    valid_destination_coordinates = matching_routes.loc[
                        destination_lat_values.notna()
                        & destination_lon_values.notna()
                        & destination_lat_values.between(5, 37)
                        & destination_lon_values.between(68, 98)
                    ].copy()

                    if not valid_destination_coordinates.empty:
                        destination_lat = float(
                            pd.to_numeric(
                                valid_destination_coordinates["destination_lat"],
                                errors="coerce"
                            ).median()
                        )
                        destination_lon = float(
                            pd.to_numeric(
                                valid_destination_coordinates["destination_lon"],
                                errors="coerce"
                            ).median()
                        )
                        destination_coordinates = (
                            destination_lat,
                            destination_lon
                        )
                        destination_from_dataset = True

                # If the selected route has no valid dataset coordinates, use the
                # destination city's coordinate only as a fallback.
                if destination_coordinates is None:
                    destination_coordinates = smartlogix_get_city_coordinates(
                        selected_map_destination
                    )

                if origin_coordinates and destination_coordinates:

                    if destination_from_dataset:
                        st.caption(
                            "📍 Destination coordinates are taken from the ORDERS dataset."
                        )
                    else:
                        st.warning(
                            "⚠️ Valid destination coordinates were not available for this route; "
                            "destination city coordinates are being used as a fallback."
                        )

                    map_col1, map_col2, map_col3 = st.columns(3)

                    with map_col1:
                        st.metric(
                            "📍 Origin",
                            selected_map_origin
                        )

                    with map_col2:
                        st.metric(
                            "🏁 Destination",
                            selected_map_destination
                        )

                    # Try to obtain the actual road route from OSRM.
                    route_api_success = False
                    route_geometry = None
                    route_distance_km = None
                    route_duration_hours = None

                    try:
                        origin_lat, origin_lon = origin_coordinates
                        destination_lat, destination_lon = destination_coordinates

                        osrm_url = (
                            "https://router.project-osrm.org/route/v1/driving/"
                            f"{origin_lon},{origin_lat};"
                            f"{destination_lon},{destination_lat}"
                            "?overview=full&geometries=geojson"
                        )

                        osrm_response = requests.get(
                            osrm_url,
                            timeout=10
                        )

                        if osrm_response.ok:
                            osrm_data = osrm_response.json()

                            if osrm_data.get("routes"):
                                selected_route = osrm_data["routes"][0]
                                route_geometry = selected_route["geometry"]
                                route_distance_km = (
                                    selected_route.get("distance", 0) / 1000
                                )
                                route_duration_hours = (
                                    selected_route.get("duration", 0) / 3600
                                )
                                route_api_success = True

                    except Exception:
                        route_api_success = False

                    if not route_api_success:
                        # Fall back to the dataset's route distance when available.
                        fallback_distance = None

                        for distance_name in [
                            "route_distance_km",
                            "distance_km",
                            "delivery_distance_km",
                            "total_distance_km"
                        ]:
                            if distance_name in visible_route_orders_df.columns:
                                candidate_distance = pd.to_numeric(
                                    visible_route_orders_df.loc[
                                        matching_routes.index,
                                        distance_name
                                    ],
                                    errors="coerce"
                                ).dropna()

                                if not candidate_distance.empty:
                                    fallback_distance = float(
                                        candidate_distance.median()
                                    )
                                    break

                        route_distance_km = fallback_distance

                    with map_col3:
                        if route_distance_km is not None:
                            st.metric(
                                "📏 Route Distance",
                                f"{route_distance_km:,.1f} km"
                            )
                        else:
                            st.metric(
                                "📏 Route Distance",
                                "N/A"
                            )

                    try:
                        import pydeck as pdk

                        origin_lat, origin_lon = origin_coordinates
                        destination_lat, destination_lon = destination_coordinates

                        points = pd.DataFrame([
                            {
                                "name": selected_map_origin,
                                "latitude": origin_lat,
                                "longitude": origin_lon
                            },
                            {
                                "name": selected_map_destination,
                                "latitude": destination_lat,
                                "longitude": destination_lon
                            }
                        ])

                        layers = [
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=points,
                                get_position="[longitude, latitude]",
                                get_radius=12000,
                                get_fill_color=[30, 144, 255, 220],
                                get_line_color=[0, 70, 140, 255],
                                line_width_min_pixels=2,
                                pickable=True
                            )
                        ]

                        # Draw the actual road route in BLUE when OSRM returns geometry.
                        if route_geometry and route_geometry.get("coordinates"):
                            route_geojson = {
                                "type": "Feature",
                                "properties": {},
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": route_geometry["coordinates"]
                                }
                            }

                            layers.append(
                                pdk.Layer(
                                    "GeoJsonLayer",
                                    data=route_geojson,
                                    stroked=True,
                                    filled=False,
                                    get_line_color=[0, 102, 255, 255],
                                    get_line_width=6,
                                    line_width_min_pixels=5,
                                    pickable=True
                                )
                            )
                        else:
                            # If routing service is unavailable, show a blue straight route line.
                            straight_line_geojson = {
                                "type": "Feature",
                                "properties": {},
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": [
                                        [origin_lon, origin_lat],
                                        [destination_lon, destination_lat]
                                    ]
                                }
                            }

                            layers.append(
                                pdk.Layer(
                                    "GeoJsonLayer",
                                    data=straight_line_geojson,
                                    stroked=True,
                                    filled=False,
                                    get_line_color=[0, 102, 255, 255],
                                    get_line_width=6,
                                    line_width_min_pixels=5,
                                    pickable=True
                                )
                            )

                        midpoint_lat = (origin_lat + destination_lat) / 2
                        midpoint_lon = (origin_lon + destination_lon) / 2

                        # Choose a useful zoom level based on the selected route span.
                        lat_span = abs(origin_lat - destination_lat)
                        lon_span = abs(origin_lon - destination_lon)
                        route_span = max(lat_span, lon_span)

                        if route_span > 30:
                            map_zoom = 4.0
                        elif route_span > 20:
                            map_zoom = 4.5
                        elif route_span > 10:
                            map_zoom = 5.0
                        elif route_span > 5:
                            map_zoom = 5.8
                        else:
                            map_zoom = 7.0

                        deck = pdk.Deck(
                            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                            initial_view_state=pdk.ViewState(
                                latitude=midpoint_lat,
                                longitude=midpoint_lon,
                                zoom=map_zoom,
                                pitch=0
                            ),
                            layers=layers,
                            tooltip={"text": "{name}"}
                        )

                        st.pydeck_chart(
                            deck,
                            width="stretch",
                            height=500
                        )

                        if route_api_success and route_duration_hours is not None:
                            st.success(
                                f"🛣️ Blue line = road route • "
                                f"Estimated driving time: {route_duration_hours:.1f} hours. "
                                f"Distance: {route_distance_km:,.1f} km."
                            )
                        else:
                            st.info(
                                "ℹ️ The routing service was unavailable, so the blue line shows a direct route between the selected cities."
                            )

                    except ImportError:
                        map_points = pd.DataFrame({
                            "latitude": [
                                origin_coordinates[0],
                                destination_coordinates[0]
                            ],
                            "longitude": [
                                origin_coordinates[1],
                                destination_coordinates[1]
                            ]
                        })

                        st.map(
                            map_points,
                            zoom=4
                        )

                        st.info(
                            "ℹ️ Install pydeck to display the connected route line on the map."
                        )

                else:
                    st.warning(
                        "⚠️ Coordinates are not available for the selected city/cities. "
                        "Add their coordinates to the SMARTLOGIX_CITY_COORDINATES dictionary."
                    )

            else:
                st.info(
                    "ℹ️ Origin and destination data are not available for the map."
                )

        else:
            st.info(
                "ℹ️ The current dataset does not contain recognizable origin and destination columns."
            )


        st.subheader("🚚 Routes by Transport Mode")

        if "transport_mode" in visible_route_orders_df.columns:

            transport_summary = (
                visible_route_orders_df["transport_mode"]
                .astype(str)
                .str.title()
                .value_counts()
                .reset_index()
            )

            transport_summary.columns = [
                "Transport Mode",
                "Orders"
            ]

            fig = px.bar(
                transport_summary,
                x="Transport Mode",
                y="Orders",
                title="Orders Assigned to Each Transport Mode"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        else:

            st.info(
                "ℹ️ Transport mode information is not available."
            )


        st.subheader("📋 Route Records")

        route_columns = [
            "order_id",
            "origin",
            "source",
            "pickup_location",
            "destination",
            "delivery_location",
            "transport_mode",
            "vehicle_id",
            "route_distance_km",
            "distance_km",
            "delivery_distance_km",
            "total_distance_km"
        ]

        available_columns = [
            column
            for column in route_columns
            if column in visible_route_orders_df.columns
        ]

        if available_columns:

            route_table = visible_route_orders_df[
                available_columns
            ].copy()

            st.dataframe(
                route_table,
                width="stretch",
                hide_index=True,
                height=500
            )

        else:

            st.info(
                "ℹ️ Route-related columns are not available."
            )

## 🚁 Drone Management

if page == "🚁 Drone Management":

    st.title("🚁 Drone Management")

    st.caption(
        "Monitor drone deliveries, assignments and operational utilization"
    )


    st.subheader("🔍 Search Drone Delivery")

    drone_order_search = st.text_input(
        "🆔 Enter Drone Order ID",
        placeholder="Example: ORD-005379",
        key="drone_order_search"
    )


    if orders_df.empty:

        st.warning("⚠️ No order data available.")

    else:

        drone_orders = orders_df.copy()

        if "transport_mode" in drone_orders.columns:

            transport_series = pd.Series(
                drone_orders["transport_mode"],
                index=drone_orders.index
            ).astype(str)

            drone_orders = drone_orders[
                transport_series.str.lower().str.contains(
                    "drone",
                    na=False
                )
            ].copy()

        else:

            drone_orders = pd.DataFrame()

        st.subheader("📊 Drone Operations Overview")

        drone_delivery_count = len(drone_orders)

        drone_count = 0

        if (
            not drone_orders.empty
            and "vehicle_id" in drone_orders.columns
        ):

            vehicle_series = pd.Series(
                drone_orders["vehicle_id"],
                index=drone_orders.index
            )

            drone_count = vehicle_series.nunique()

        active_drone_deliveries = 0

        if (
            not drone_orders.empty
            and "order_status" in drone_orders.columns
        ):

            status_series = pd.Series(
                drone_orders["order_status"],
                index=drone_orders.index
            ).astype(str)

            active_drone_deliveries = drone_orders[
                status_series.str.lower().isin(
                    [
                        "in transit",
                        "out for delivery",
                        "dispatched"
                    ]
                )
            ].shape[0]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🚁 Drone Fleet",
                f"{drone_count:,}"
            )

        with col2:

            st.metric(
                "📦 Drone Deliveries",
                f"{drone_delivery_count:,}"
            )

        with col3:

            st.metric(
                "📍 Active Drone Deliveries",
                f"{active_drone_deliveries:,}"
            )


        if drone_orders.empty:

            st.info(
                "ℹ️ No drone delivery records were found."
            )

        else:

            if drone_order_search:

                drone_order_search = drone_order_search.strip()

                order_id_series = pd.Series(
                    drone_orders["order_id"],
                    index=drone_orders.index
                ).astype(str)

                search_result = drone_orders[
                    order_id_series.str.upper().eq(
                        drone_order_search.upper()
                    )
                ]

                if search_result.empty:

                    st.error(
                        f"❌ Drone order {drone_order_search} was not found."
                    )

                else:

                    drone_order = search_result.iloc[0]

                    st.success(
                        f"✅ Drone delivery {drone_order_search} found"
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "📦 Order ID",
                            str(
                                drone_order.get(
                                    "order_id",
                                    "N/A"
                                )
                            )
                        )

                    with col2:

                        st.metric(
                            "📍 Status",
                            str(
                                drone_order.get(
                                    "order_status",
                                    "N/A"
                                )
                            ).title()
                        )

                    with col3:

                        st.metric(
                            "🚁 Vehicle ID",
                            str(
                                drone_order.get(
                                    "vehicle_id",
                                    "N/A"
                                )
                            )
                        )

                    with col4:

                        st.metric(
                            "🚐 Vehicle Type",
                            str(
                                drone_order.get(
                                    "vehicle_type",
                                    "N/A"
                                )
                            )
                        )


                    st.subheader("📋 Drone Delivery Details")

                    detail_columns = [
                        "order_id",
                        "order_status",
                        "transport_mode",
                        "vehicle_id",
                        "vehicle_type",
                        "vehicle_count",
                        "vehicle_avg_speed_kmph",
                        "vehicle_capacity_utilization_pct"
                    ]

                    available_detail_columns = [
                        column
                        for column in detail_columns
                        if column in search_result.columns
                    ]

                    st.dataframe(
                        search_result[
                            available_detail_columns
                        ].T.rename(
                            columns={
                                search_result.index[0]: "Value"
                            }
                        ),
                        width="stretch"
                    )


            st.subheader("📈 Drone Delivery Status")

            if "order_status" in drone_orders.columns:

                status_series = pd.Series(
                    drone_orders["order_status"],
                    index=drone_orders.index
                ).astype(str)

                status_summary = (
                    status_series
                    .str.title()
                    .value_counts()
                    .reset_index()
                )

                status_summary.columns = [
                    "Delivery Status",
                    "Orders"
                ]

                fig = px.bar(
                    status_summary,
                    x="Delivery Status",
                    y="Orders",
                    title="Drone Deliveries by Status"
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )


            st.subheader("🚁 Drone Assignment Records")

            drone_columns = [
                "order_id",
                "order_status",
                "transport_mode",
                "vehicle_id",
                "vehicle_type",
                "vehicle_count",
                "vehicle_avg_speed_kmph",
                "vehicle_capacity_utilization_pct"
            ]

            available_columns = [
                column
                for column in drone_columns
                if column in drone_orders.columns
            ]

            st.dataframe(
                drone_orders[
                    available_columns
                ],
                width="stretch",
                hide_index=True,
                height=500
            )

## 📊 Analytics

if page == "🛠️ Predictive Maintenance":

    st.title("🛠️ Predictive Maintenance")
    st.caption("Vehicle-level maintenance risk prediction using the trained Random Forest model")

    fleet_path = "cleaned_data/fleet_vehicles_cleaned.csv"
    delivery_log_path = "cleaned_data/delivery_logs_clean.csv"

    if maintenance_model is None:
        st.error("maintenance_model.pkl could not be loaded from the models folder.")

    elif not os.path.exists(fleet_path):
        st.error("Fleet data file was not found: cleaned_data/fleet_vehicles_cleaned.csv")

    else:
        fleet_df = pd.read_csv(fleet_path)
        logs_df = pd.read_csv(delivery_log_path) if os.path.exists(delivery_log_path) else pd.DataFrame()

        if "vehicle_id" not in fleet_df.columns or fleet_df.empty:
            st.warning("Fleet vehicle data is unavailable.")

        else:
            vehicle_ids = (
                fleet_df["vehicle_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_vehicle = st.selectbox(
                "Select Vehicle",
                vehicle_ids
            )

            vehicle = fleet_df[
                fleet_df["vehicle_id"].astype(str) == selected_vehicle
            ].iloc[0]

            vehicle_type = str(
                vehicle.get("vehicle_type", "Unknown")
            ).strip()

            capacity = pd.to_numeric(
                vehicle.get("capacity_kg"),
                errors="coerce"
            )

            max_range = pd.to_numeric(
                vehicle.get("max_range_km"),
                errors="coerce"
            )

            avg_speed = pd.to_numeric(
                vehicle.get("vehicle_avg_speed_kmph", vehicle.get("avg_speed_kmph")),
                errors="coerce"
            )

            ownership = vehicle.get("ownership", 0)

            vehicle_orders = pd.DataFrame()

            if not orders_df.empty and "vehicle_id" in orders_df.columns:
                vehicle_orders = orders_df[
                    orders_df["vehicle_id"].astype(str).eq(selected_vehicle)
                ].copy()

            delivery_event_count = 0
            failed_attempt_count = 0
            rto_count = 0
            hub_event_count = 0

            if not logs_df.empty and "vehicle_id" in logs_df.columns:
                vehicle_logs = logs_df[
                    logs_df["vehicle_id"].astype(str).eq(selected_vehicle)
                ].copy()

                delivery_event_count = len(vehicle_logs)

                if "event_type" in vehicle_logs.columns:
                    event_text = (
                        vehicle_logs["event_type"]
                        .astype(str)
                        .str.upper()
                    )

                    failed_attempt_count = int(
                        event_text.str.contains(
                            "FAILED|ATTEMPT",
                            regex=True,
                            na=False
                        ).sum()
                    )

                    rto_count = int(
                        event_text.str.contains(
                            "RTO|RETURN",
                            regex=True,
                            na=False
                        ).sum()
                    )

                    hub_event_count = int(
                        event_text.str.contains(
                            "HUB|AT_HUB",
                            regex=True,
                            na=False
                        ).sum()
                    )

            utilization = pd.to_numeric(
                vehicle.get("vehicle_capacity_utilization_pct"),
                errors="coerce"
            )

            if (
                pd.isna(utilization)
                and not vehicle_orders.empty
                and pd.notna(capacity)
                and capacity > 0
                and "package_weight" in vehicle_orders.columns
            ):
                weights = pd.to_numeric(
                    vehicle_orders["package_weight"],
                    errors="coerce"
                )

                utilization_values = (
                    weights / capacity * 100
                ).dropna()

                if not utilization_values.empty:
                    utilization = float(
                        utilization_values.median()
                    )

            weight_exceeds_capacity = 0

            if (
                not vehicle_orders.empty
                and "package_weight" in vehicle_orders.columns
                and pd.notna(capacity)
                and capacity > 0
            ):
                weights = pd.to_numeric(
                    vehicle_orders["package_weight"],
                    errors="coerce"
                )

                weight_exceeds_capacity = int(
                    (weights > capacity).sum() > 0
                )

            distance_km = 0.0

            if (
                not vehicle_orders.empty
                and "distance_km" in vehicle_orders.columns
            ):
                distances = pd.to_numeric(
                    vehicle_orders["distance_km"],
                    errors="coerce"
                ).dropna()

                if not distances.empty:
                    distance_km = float(
                        distances.median()
                    )

            # -----------------------------------------------------
            # READ THE NEW MAINTENANCE MODEL PACKAGE
            # -----------------------------------------------------

            if isinstance(maintenance_model, dict):
                maintenance_rf = maintenance_model.get("model")

                maintenance_features = maintenance_model.get(
                    "features",
                    [
                        "vehicle_type",
                        "capacity_kg",
                        "max_range_km",
                        "vehicle_avg_speed_kmph",
                        "ownership",
                        "vehicle_capacity_utilization_pct",
                        "weight_exceeds_capacity",
                        "distance_km",
                        "delivery_event_count",
                        "failed_attempt_count",
                        "rto_count",
                        "hub_event_count"
                    ]
                )

                vehicle_encoder = (
                    maintenance_model.get("vehicle_type_encoder")
                    or maintenance_model.get("le_vehicle_type")
                    or maintenance_model.get("vehicle_type_le")
                )

                ownership_encoder = (
                    maintenance_model.get("ownership_encoder")
                    or maintenance_model.get("le_ownership")
                    or maintenance_model.get("ownership_le")
                )

                vehicle_type_mapping = maintenance_model.get(
                    "vehicle_type_mapping"
                )

                ownership_mapping = maintenance_model.get(
                    "ownership_mapping"
                )

            else:
                maintenance_rf = maintenance_model
                maintenance_features = [
                    "vehicle_type",
                    "capacity_kg",
                    "max_range_km",
                    "vehicle_avg_speed_kmph",
                    "ownership",
                    "vehicle_capacity_utilization_pct",
                    "weight_exceeds_capacity",
                    "distance_km",
                    "delivery_event_count",
                    "failed_attempt_count",
                    "rto_count",
                    "hub_event_count"
                ]
                vehicle_encoder = None
                ownership_encoder = None
                vehicle_type_mapping = None
                ownership_mapping = None

            if maintenance_rf is None:
                st.error("The maintenance model package does not contain a trained model.")
                st.stop()

            # -----------------------------------------------------
            # ENCODE VEHICLE TYPE EXACTLY AS TRAINING
            # -----------------------------------------------------

            try:
                if vehicle_encoder is not None:
                    encoded_vehicle_type = int(
                        vehicle_encoder.transform([vehicle_type])[0]
                    )

                elif isinstance(vehicle_type_mapping, dict):
                    if vehicle_type not in vehicle_type_mapping:
                        raise ValueError(
                            f"Vehicle type '{vehicle_type}' is not present in the saved encoder."
                        )
                    encoded_vehicle_type = int(
                        vehicle_type_mapping[vehicle_type]
                    )

                else:
                    default_vehicle_mapping = {
                        "Air Cargo": 0,
                        "Bike": 1,
                        "Drone": 2,
                        "Ship": 3,
                        "Truck": 4,
                        "Van": 5
                    }

                    if vehicle_type not in default_vehicle_mapping:
                        raise ValueError(
                            f"Unknown vehicle type: {vehicle_type}"
                        )

                    encoded_vehicle_type = int(
                        default_vehicle_mapping[vehicle_type]
                    )

            except Exception as e:
                st.error(
                    f"Vehicle type encoding failed: {e}"
                )
                st.stop()

            # -----------------------------------------------------
            # ENCODE OWNERSHIP USING THE EXACT TRAINING MAPPING
            # -----------------------------------------------------
            # Original training data mapping:
            # 3PL Partner -> 0
            # Leased      -> 1
            # Owned       -> 2
            #
            # We use this fixed mapping because the saved ownership
            # encoder was created after the ownership column had already
            # been converted to numeric values.
            # -----------------------------------------------------

            try:
                ownership_mapping_exact = {
                    "3PL Partner": 0,
                    "Leased": 1,
                    "Owned": 2
                }

                ownership_text = str(ownership).strip()

                if ownership_text in ownership_mapping_exact:
                    encoded_ownership = ownership_mapping_exact[
                        ownership_text
                    ]

                else:
                    # Also allow already-encoded numeric ownership values.
                    ownership_numeric = pd.to_numeric(
                        ownership,
                        errors="coerce"
                    )

                    if pd.isna(ownership_numeric):
                        raise ValueError(
                            f"Ownership value '{ownership}' is not recognised. "
                            f"Expected: {list(ownership_mapping_exact.keys())}"
                        )

                    encoded_ownership = int(ownership_numeric)

                    if encoded_ownership not in [0, 1, 2]:
                        raise ValueError(
                            f"Encoded ownership value '{encoded_ownership}' "
                            f"is outside the training range 0, 1, 2."
                        )

            except Exception as e:
                st.error(
                    f"Ownership encoding failed: {e}"
                )
                st.stop()

            # -----------------------------------------------------
            # CREATE EXACT 12 MODEL FEATURES
            # -----------------------------------------------------

            values = {
                "vehicle_type": int(encoded_vehicle_type),
                "capacity_kg": 0.0 if pd.isna(capacity) else float(capacity),
                "max_range_km": 0.0 if pd.isna(max_range) else float(max_range),
                "vehicle_avg_speed_kmph": 0.0 if pd.isna(avg_speed) else float(avg_speed),
                "ownership": int(encoded_ownership),
                "vehicle_capacity_utilization_pct": 0.0 if pd.isna(utilization) else float(utilization),
                "weight_exceeds_capacity": int(weight_exceeds_capacity),
                "distance_km": float(distance_km),
                "delivery_event_count": int(delivery_event_count),
                "failed_attempt_count": int(failed_attempt_count),
                "rto_count": int(rto_count),
                "hub_event_count": int(hub_event_count)
            }

            try:
                X_maint = pd.DataFrame(
                    [[values[feature] for feature in maintenance_features]],
                    columns=maintenance_features
                )
            except KeyError as e:
                st.error(
                    f"Maintenance feature mismatch: {e}"
                )
                st.stop()

            # -----------------------------------------------------
            # PREDICT
            # -----------------------------------------------------

            try:
                prediction = int(
                    maintenance_rf.predict(X_maint)[0]
                )

                probability = None

                if hasattr(maintenance_rf, "predict_proba"):
                    probabilities = maintenance_rf.predict_proba(X_maint)[0]

                    if hasattr(maintenance_rf, "classes_"):
                        classes = list(maintenance_rf.classes_)

                        if 1 in classes:
                            probability = float(
                                probabilities[classes.index(1)]
                            )
                        else:
                            probability = float(
                                probabilities.max()
                            )
                    else:
                        probability = float(
                            probabilities[-1]
                        )

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric(
                        "Vehicle",
                        selected_vehicle
                    )

                with c2:
                    st.metric(
                        "Type",
                        vehicle_type
                    )

                with c3:
                    st.metric(
                        "Maintenance Events",
                        f"{delivery_event_count:,}"
                    )

                with c4:
                    st.metric(
                        "Capacity",
                        f"{values['capacity_kg']:.1f} kg"
                    )

                if prediction == 1:
                    st.warning(
                        "⚠️ Maintenance Required"
                    )
                else:
                    st.success(
                        "✅ No Maintenance Required"
                    )

                if probability is not None:
                    st.metric(
                        "Model Risk Probability",
                        f"{probability:.1%}"
                    )

                st.subheader(
                    "Maintenance Model Inputs"
                )

                display_values = values.copy()
                display_values["vehicle_type"] = vehicle_type
                display_values["ownership"] = ownership

                maintenance_display = pd.DataFrame(
                    {
                        "Feature": maintenance_features,
                        "Value": [
                            display_values[feature]
                            for feature in maintenance_features
                        ]
                    }
                )

                st.dataframe(
                    maintenance_display,
                    width="stretch",
                    hide_index=True
                )

                st.info(
                    "This is a vehicle/fleet-level predictive-maintenance result. "
                    "It is separate from drone image damage detection."
                )

            except Exception as e:
                st.error(
                    f"Maintenance prediction could not be calculated: {e}"
                )

if page == "📈 Analytics":

    st.title("📊 Analytics")

    st.caption(
        "Explore logistics performance, delivery trends and operational data"
    )


    if orders_df.empty:

        st.warning("⚠️ No order data available.")

    else:

        st.subheader("📦 Order Overview")

        total_orders = len(orders_df)

        delivered_orders = 0
        in_transit_orders = 0
        delayed_orders = 0

        if "order_status" in orders_df.columns:

            status_series = pd.Series(
                orders_df["order_status"],
                index=orders_df.index
            ).astype(str).str.lower()

            delivered_orders = status_series.eq(
                "delivered"
            ).sum()

            in_transit_orders = status_series.isin(
                [
                    "in transit",
                    "out for delivery",
                    "dispatched"
                ]
            ).sum()

            delayed_orders = int(
                pd.to_numeric(
                    orders_df["delivery_delay_hours"],
                    errors="coerce"
                ).gt(0).sum()
            )

        delivery_rate = 0

        if total_orders > 0:

            delivery_rate = (
                delivered_orders / total_orders
            ) * 100

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "📦 Total Orders",
                f"{total_orders:,}"
            )

        with col2:

            st.metric(
                "✅ Delivered",
                f"{delivered_orders:,}"
            )

        with col3:

            st.metric(
                "🚚 In Transit",
                f"{in_transit_orders:,}"
            )

        with col4:

            st.metric(
                "📈 Delivery Rate",
                f"{delivery_rate:.1f}%"
            )


        st.subheader("📍 Delivery Status Analysis")

        if "order_status" in orders_df.columns:

            status_series = pd.Series(
                orders_df["order_status"],
                index=orders_df.index
            ).astype(str)

            status_summary = (
                status_series
                .str.title()
                .value_counts()
                .reset_index()
            )

            status_summary.columns = [
                "Delivery Status",
                "Orders"
            ]

            col1, col2 = st.columns(2)

            with col1:

                fig = px.bar(
                    status_summary,
                    x="Delivery Status",
                    y="Orders",
                    title="Orders by Delivery Status"
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

            with col2:

                fig = px.pie(
                    status_summary,
                    names="Delivery Status",
                    values="Orders",
                    title="Delivery Status Distribution"
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )


        st.subheader("🚚 Transport Mode Analysis")

        if "transport_mode" in orders_df.columns:

            transport_series = pd.Series(
                orders_df["transport_mode"],
                index=orders_df.index
            ).astype(str)

            transport_summary = (
                transport_series
                .str.title()
                .value_counts()
                .reset_index()
            )

            transport_summary.columns = [
                "Transport Mode",
                "Orders"
            ]

            fig = px.bar(
                transport_summary,
                x="Transport Mode",
                y="Orders",
                title="Orders by Transport Mode"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )


        st.subheader("🚛 Fleet & Delivery Metrics")

        vehicle_count = 0
        average_speed = None
        average_utilization = None

        if "vehicle_id" in orders_df.columns:

            vehicle_count = orders_df[
                "vehicle_id"
            ].nunique()

        if "vehicle_avg_speed_kmph" in orders_df.columns:

            speed_series = pd.to_numeric(
                orders_df["vehicle_avg_speed_kmph"],
                errors="coerce"
            )

            average_speed = speed_series.mean()

        if "vehicle_capacity_utilization_pct" in orders_df.columns:

            utilization_series = pd.to_numeric(
                orders_df[
                    "vehicle_capacity_utilization_pct"
                ],
                errors="coerce"
            )

            average_utilization = utilization_series.mean()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🚛 Vehicles Assigned",
                f"{vehicle_count:,}"
            )

        with col2:

            if pd.notna(average_speed):

                st.metric(
                    "⚡ Average Vehicle Speed",
                    f"{average_speed:.1f} km/h"
                )

            else:

                st.metric(
                    "⚡ Average Vehicle Speed",
                    "N/A"
                )

        with col3:

            if pd.notna(average_utilization):

                st.metric(
                    "📊 Avg Capacity Utilization",
                    f"{average_utilization:.1f}%"
                )

            else:

                st.metric(
                    "📊 Avg Capacity Utilization",
                    "N/A"
                )


        st.subheader("📋 Analytics Data")

        analytics_columns = [
            "order_id",
            "order_status",
            "transport_mode",
            "vehicle_id",
            "vehicle_type",
            "vehicle_count",
            "vehicle_avg_speed_kmph",
            "vehicle_capacity_utilization_pct"
        ]

        available_columns = [
            column
            for column in analytics_columns
            if column in orders_df.columns
        ]

        st.dataframe(
            orders_df[
                available_columns
            ],
            width="stretch",
            hide_index=True,
            height=500
        )

# ML PREDICTIONS
if page == "🤖 ML Predictions":

    st.header("🤖 ML Predictions")
    st.caption("AI-powered transport mode selection and delivery ETA prediction")

    st.subheader("📦 New Order Prediction")
    st.info("Enter the basic order details. SmartLogix automatically derives distance and supporting ETA features from historical logistics data.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        quantity = st.number_input(
            "📦 Quantity",
            min_value=1,
            value=1,
            step=1
        )

    with col2:
        package_weight = st.number_input(
            "⚖️ Package Weight (kg)",
            min_value=0.01,
            value=1.0,
            step=0.1
        )

    with col3:
        order_value_inr = st.number_input(
            "💰 Order Value (₹)",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

    with col4:
        delivery_priority = st.selectbox(
            "🚨 Delivery Priority",
            ["economy", "standard", "express", "same-day"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        customer_segment = st.selectbox(
            "👤 Customer Segment",
            ["Business", "Enterprise", "Retail", "SMB"]
        )

    with col2:
        origin_options = (
            sorted(
                orders_df["origin_city"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if "origin_city" in orders_df.columns
            else []
        )

        origin_city = (
            st.selectbox(
                "📍 Origin City",
                origin_options
            )
            if origin_options
            else None
        )

    with col3:
        destination_options = (
            sorted(
                orders_df["destination_city"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if "destination_city" in orders_df.columns
            else []
        )

        destination_city = (
            st.selectbox(
                "📍 Destination City",
                destination_options
            )
            if destination_options
            else None
        )

    predict_button = st.button(
        "🚀 Predict Transport & ETA",
        type="primary",
        width="stretch"
    )

    if predict_button:

        try:
            if not origin_city or not destination_city:
                st.error("Origin and destination city data are unavailable.")
                st.stop()

            if not isinstance(transport_model, dict):
                st.error(
                    "The transport model file is not in the new 8-feature model-package format. "
                    "Please copy the newly trained random_forest.pkl into the models folder."
                )
                st.stop()

            rf_model = transport_model["model"]
            transport_encoders = transport_model["encoders"]
            transport_features = transport_model["features"]

            expected_transport_features = [
                "quantity",
                "distance_km",
                "package_weight",
                "order_value_inr",
                "delivery_priority",
                "customer_segment",
                "origin_city",
                "destination_city"
            ]

            if transport_features != expected_transport_features:
                st.error(
                    "The saved transport model features do not match the Streamlit input features."
                )
                st.write("Saved features:", transport_features)
                st.write("Expected features:", expected_transport_features)
                st.stop()

            # -------------------------------------------------
            # AUTOMATIC DISTANCE FROM HISTORICAL CITY PAIR
            # -------------------------------------------------

            origin_clean = str(origin_city).strip().lower()
            destination_clean = str(destination_city).strip().lower()

            pair_rows = orders_df.copy()

            pair_rows["_origin_clean"] = (
                pair_rows["origin_city"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            pair_rows["_destination_clean"] = (
                pair_rows["destination_city"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            pair_rows = pair_rows[
                (pair_rows["_origin_clean"] == origin_clean)
                &
                (pair_rows["_destination_clean"] == destination_clean)
            ]

            distance_km = None

            if "distance_km" in pair_rows.columns and not pair_rows.empty:
                pair_distances = pd.to_numeric(
                    pair_rows["distance_km"],
                    errors="coerce"
                ).dropna()

                if not pair_distances.empty:
                    distance_km = float(
                        pair_distances.median()
                    )

            if distance_km is None and "distance_km" in orders_df.columns:
                destination_distance_rows = orders_df[
                    orders_df["destination_city"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq(destination_clean)
                ]

                destination_distances = pd.to_numeric(
                    destination_distance_rows["distance_km"],
                    errors="coerce"
                ).dropna()

                if not destination_distances.empty:
                    distance_km = float(
                        destination_distances.median()
                    )

            if distance_km is None and "distance_km" in orders_df.columns:
                all_distances = pd.to_numeric(
                    orders_df["distance_km"],
                    errors="coerce"
                ).dropna()

                if not all_distances.empty:
                    distance_km = float(
                        all_distances.median()
                    )

            if distance_km is None:
                raise ValueError(
                    "Distance could not be derived from the historical logistics data."
                )

            # -------------------------------------------------
            # TRANSPORT MODEL INPUT
            # -------------------------------------------------

            transport_input = pd.DataFrame([
                {
                    "quantity": quantity,
                    "distance_km": distance_km,
                    "package_weight": package_weight,
                    "order_value_inr": order_value_inr,
                    "delivery_priority": delivery_priority,
                    "customer_segment": customer_segment,
                    "origin_city": origin_city,
                    "destination_city": destination_city
                }
            ])

            for column in [
                "delivery_priority",
                "customer_segment",
                "origin_city",
                "destination_city"
            ]:

                if column not in transport_encoders:
                    raise ValueError(
                        f"Saved encoder for '{column}' was not found in random_forest.pkl."
                    )

                encoder = transport_encoders[column]
                value = str(
                    transport_input[column].iloc[0]
                )

                if value not in encoder.classes_:
                    raise ValueError(
                        f"Value '{value}' for '{column}' was not present in the training data."
                    )

                transport_input[column] = encoder.transform(
                    [value]
                )

            transport_input = transport_input[
                transport_features
            ]

            predicted_mode = str(
                rf_model.predict(
                    transport_input
                )[0]
            )

            probabilities = rf_model.predict_proba(
                transport_input
            )[0]

            confidence = float(
                probabilities.max() * 100
            )

            st.success(
                f"### 🚚 Recommended Transport Mode: {predicted_mode.title()}"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Predicted Mode",
                    predicted_mode.title()
                )

            with c2:
                st.metric(
                    "Model Confidence",
                    f"{confidence:.1f}%"
                )

            with c3:
                st.metric(
                    "Derived Distance",
                    f"{distance_km:.1f} km"
                )

            probability_df = pd.DataFrame(
                {
                    "Transport Mode": [
                        str(x).title()
                        for x in rf_model.classes_
                    ],
                    "Probability (%)": [
                        round(float(x) * 100, 2)
                        for x in probabilities
                    ]
                }
            ).sort_values(
                "Probability (%)",
                ascending=False
            ).reset_index(drop=True)

            st.subheader(
                "📊 Transport Mode Probabilities"
            )

            st.dataframe(
                probability_df,
                width="stretch",
                hide_index=True
            )

            # -------------------------------------------------
            # DRONE ASSESSMENT
            # -------------------------------------------------

            if predicted_mode.lower() == "drone":

                st.subheader("🚁 Drone Delivery Assessment")

                drone_rows = (
                    orders_df[
                        orders_df["transport_mode"]
                        .astype(str)
                        .str.lower()
                        .eq("drone")
                    ]
                    if "transport_mode" in orders_df.columns
                    else pd.DataFrame()
                )

                drone_capacity = None
                drone_speed = None

                if (
                    not drone_rows.empty
                    and "capacity_kg" in drone_rows.columns
                ):
                    drone_capacity_values = pd.to_numeric(
                        drone_rows["capacity_kg"],
                        errors="coerce"
                    ).dropna()

                    if not drone_capacity_values.empty:
                        drone_capacity = float(
                            drone_capacity_values.median()
                        )

                if (
                    not drone_rows.empty
                    and "vehicle_avg_speed_kmph" in drone_rows.columns
                ):
                    drone_speed_values = pd.to_numeric(
                        drone_rows["vehicle_avg_speed_kmph"],
                        errors="coerce"
                    ).dropna()

                    if not drone_speed_values.empty:
                        drone_speed = float(
                            drone_speed_values.median()
                        )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Historical Drone Capacity",
                        f"{drone_capacity:.2f} kg"
                        if drone_capacity is not None
                        else "Not available"
                    )

                    if drone_capacity is not None:
                        if package_weight <= drone_capacity:
                            st.success(
                                "✅ Payload within historical capacity"
                            )
                        else:
                            st.error(
                                "❌ Payload exceeds historical capacity"
                            )

                with c2:
                    st.metric(
                        "Historical Drone Speed",
                        f"{drone_speed:.1f} km/h"
                        if drone_speed is not None
                        else "Not available"
                    )

                with c3:
                    st.metric(
                        "Battery / Range",
                        "Not available"
                    )

                st.caption(
                    "Battery/range feasibility is not calculated because verified battery/range fields are not currently available in the connected dataset. No artificial battery result is shown."
                )

            # -------------------------------------------------
            # ETA PREDICTION
            # -------------------------------------------------

            eta_features = [
                "quantity",
                "distance_km",
                "package_weight",
                "is_fragile",
                "is_hazmat",
                "cold_chain_required",
                "delivery_priority",
                "order_value_inr",
                "origin_city",
                "destination_city",
                "destination_state",
                "weather_condition_at_dest",
                "humidity_pct",
                "precipitation_mm",
                "wind_speed_kmph",
                "visibility_km",
                "temp_celsius",
                "vehicle_type",
                "capacity_kg",
                "vehicle_avg_speed_kmph"
            ]

            missing_eta = [
                c for c in eta_features
                if c not in orders_df.columns
            ]

            if missing_eta:
                st.warning(
                    f"ETA prediction unavailable because these columns are missing: {missing_eta}"
                )

            else:
                mode_rows = orders_df[
                    orders_df["transport_mode"]
                    .astype(str)
                    .str.lower()
                    .eq(predicted_mode.lower())
                ]

                if mode_rows.empty:
                    mode_rows = orders_df

                destination_rows = orders_df[
                    orders_df["destination_city"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq(destination_clean)
                ]

                if destination_rows.empty:
                    destination_rows = orders_df

                def safe_mode_value(
                    df,
                    column,
                    fallback_df=None
                ):
                    if column in df.columns:
                        vals = (
                            df[column]
                            .dropna()
                            .astype(str)
                        )

                        if not vals.empty:
                            return vals.mode().iloc[0]

                    if (
                        fallback_df is not None
                        and column in fallback_df.columns
                    ):
                        vals = (
                            fallback_df[column]
                            .dropna()
                            .astype(str)
                        )

                        if not vals.empty:
                            return vals.mode().iloc[0]

                    return ""

                destination_state = safe_mode_value(
                    destination_rows,
                    "destination_state",
                    orders_df
                )

                vehicle_type = safe_mode_value(
                    mode_rows,
                    "vehicle_type",
                    orders_df
                )

                weather_condition = safe_mode_value(
                    destination_rows,
                    "weather_condition_at_dest",
                    orders_df
                )

                def numeric_median(
                    df,
                    column,
                    fallback_df=None,
                    fallback=0.0
                ):
                    if column in df.columns:
                        vals = pd.to_numeric(
                            df[column],
                            errors="coerce"
                        ).dropna()

                        if not vals.empty:
                            return float(
                                vals.median()
                            )

                    if (
                        fallback_df is not None
                        and column in fallback_df.columns
                    ):
                        vals = pd.to_numeric(
                            fallback_df[column],
                            errors="coerce"
                        ).dropna()

                        if not vals.empty:
                            return float(
                                vals.median()
                            )

                    return float(fallback)

                capacity_kg = numeric_median(
                    mode_rows,
                    "capacity_kg",
                    orders_df,
                    100.0
                )

                vehicle_avg_speed = numeric_median(
                    mode_rows,
                    "vehicle_avg_speed_kmph",
                    orders_df,
                    40.0
                )

                humidity = numeric_median(
                    destination_rows,
                    "humidity_pct",
                    orders_df
                )

                precipitation = numeric_median(
                    destination_rows,
                    "precipitation_mm",
                    orders_df
                )

                wind_speed = numeric_median(
                    destination_rows,
                    "wind_speed_kmph",
                    orders_df
                )

                visibility = numeric_median(
                    destination_rows,
                    "visibility_km",
                    orders_df
                )

                temperature = numeric_median(
                    destination_rows,
                    "temp_celsius",
                    orders_df
                )

                priority_mapping = {
                    "economy": 0,
                    "standard": 1,
                    "express": 2,
                    "same-day": 3
                }

                eta_input = pd.DataFrame([
                    {
                        "quantity": quantity,
                        "distance_km": distance_km,
                        "package_weight": package_weight,
                        "is_fragile": 0,
                        "is_hazmat": 0,
                        "cold_chain_required": 0,
                        "delivery_priority": priority_mapping[delivery_priority],
                        "order_value_inr": order_value_inr,
                        "origin_city": origin_city,
                        "destination_city": destination_city,
                        "destination_state": destination_state,
                        "weather_condition_at_dest": weather_condition,
                        "humidity_pct": humidity,
                        "precipitation_mm": precipitation,
                        "wind_speed_kmph": wind_speed,
                        "visibility_km": visibility,
                        "temp_celsius": temperature,
                        "vehicle_type": vehicle_type,
                        "capacity_kg": capacity_kg,
                        "vehicle_avg_speed_kmph": vehicle_avg_speed
                    }
                ])

                for column in [
                    "origin_city",
                    "destination_city",
                    "destination_state",
                    "weather_condition_at_dest",
                    "vehicle_type"
                ]:
                    encoder = LabelEncoder()
                    values = (
                        orders_df[column]
                        .dropna()
                        .astype(str)
                    )

                    if values.empty:
                        raise ValueError(
                            f"No historical values available for {column}."
                        )

                    encoder.fit(values)
                    value = str(
                        eta_input[column].iloc[0]
                    )

                    if value not in encoder.classes_:
                        value = encoder.classes_[0]

                    eta_input[column] = encoder.transform(
                        [value]
                    )

                eta_input = eta_input[
                    eta_features
                ]

                predicted_hours = max(
                    float(
                        eta_model.predict(
                            eta_input
                        )[0]
                    ),
                    0.0
                )

                st.subheader("⏱️ Estimated Delivery Time")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Predicted ETA",
                        f"{predicted_hours:.1f} hours"
                    )

                with c2:
                    st.metric(
                        "Transport Mode",
                        predicted_mode.title()
                    )

                with c3:
                    st.metric(
                        "Vehicle Type",
                        vehicle_type.title()
                    )

                with st.expander(
                    "🔍 View Automatically Derived ML Inputs"
                ):
                    derived_data = pd.DataFrame(
                        {
                            "Feature": [
                                "Distance",
                                "Destination State",
                                "Fragile",
                                "Hazmat",
                                "Cold Chain",
                                "Vehicle Type",
                                "Vehicle Capacity",
                                "Vehicle Speed",
                                "Weather"
                            ],
                            "Value": [
                                f"{distance_km:.1f} km",
                                destination_state,
                                "No",
                                "No",
                                "No",
                                vehicle_type,
                                round(float(capacity_kg), 2),
                                round(float(vehicle_avg_speed), 2),
                                weather_condition
                            ]
                        }
                    )

                    st.dataframe(
                        derived_data,
                        width="stretch",
                        hide_index=True
                    )

                st.caption(
                    "ETA is generated by the trained Random Forest regression model using the model's required logistics and weather features."
                )

        except Exception as e:
            st.error(
                f"❌ Prediction failed: {e}"
            )

## 🔍 Drone Damage Detection

if page == "🔍 Drone Damage Detection":

    st.title("🔍 Drone Damage Detection")
    st.caption("YOLO-based computer vision for inspecting drone condition from uploaded images")

    expected_drone_classes = ["Healthy Drone", "Damage Drone"]

    st.info(
        "**Computer Vision Model:** YOLO11n • **Classes:** Healthy Drone and Damage Drone • "
        "Upload a drone image to inspect its visible condition."
    )

    if YOLO is None:
        st.error("Ultralytics is not installed. Install it with: `pip install ultralytics`")
        st.stop()

    if not os.path.exists(DRONE_MODEL_PATH):
        st.error("Drone model not found. Place **best.pt** inside the **models** folder.")
        st.stop()

    if drone_detection_model is None:
        st.error(
            "The drone model could not be loaded. Verify that **models/best.pt** is a valid Ultralytics YOLO model."
        )
        st.stop()

    model_names = getattr(drone_detection_model, "names", {}) or {}

    if isinstance(model_names, dict):
        actual_class_names = [str(model_names[k]) for k in sorted(model_names.keys())]
    else:
        actual_class_names = [str(x) for x in model_names]


    upload_col, prediction_col = st.columns([1, 1.35], gap="large")

    # ==========================================================
    # LEFT SIDE - IMAGE UPLOAD
    # ==========================================================

    with upload_col:
        st.subheader("📤 Upload Drone Image")

        uploaded_drone_image = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear drone image for computer-vision inspection.",
            label_visibility="collapsed",
        )

        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Minimum confidence required for a YOLO detection to be displayed.",
        )

        input_image = None
        detect_button = False

        if uploaded_drone_image is not None:
            try:
                image_bytes = uploaded_drone_image.getvalue()
                input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                st.error(f"Unable to read the uploaded image: {e}")

            if input_image is not None:
                st.caption(f"📄 {uploaded_drone_image.name}")
                st.image(input_image, width="stretch")

                detect_button = st.button(
                    "🔎 Run AI Inspection",
                    type="primary",
                    width="stretch",
                )
        else:
            st.caption("Supported formats: JPG, JPEG, PNG, WEBP")

    # ==========================================================
    # RIGHT SIDE - PREDICTION
    # ==========================================================

    with prediction_col:
        st.subheader("🎯 AI Prediction")

        if uploaded_drone_image is None or input_image is None:
            st.info(
                "Upload a drone image to view the AI prediction, confidence score, "
                "and annotated detection."
            )

        elif not detect_button:
            st.info(
                "Your image is ready for inspection. Click **Run AI Inspection** "
                "to generate the prediction."
            )

        else:
            try:
                with st.spinner("Running YOLO11n inspection..."):
                    results = drone_detection_model.predict(
                        source=input_image,
                        conf=confidence_threshold,
                        verbose=False,
                    )

                if not results:
                    st.warning("Inspection completed, but no prediction result was returned.")

                else:
                    result = results[0]
                    names = getattr(result, "names", {}) or model_names
                    detections = []

                    # --------------------------------------------------
                    # Extract YOLO detections
                    # --------------------------------------------------

                    if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
                        boxes = result.boxes

                        class_ids = (
                            boxes.cls.detach().cpu().numpy().astype(int)
                        )
                        confidences = boxes.conf.detach().cpu().numpy()

                        for class_id, conf in zip(class_ids, confidences):
                            if isinstance(names, dict):
                                class_name = names.get(int(class_id), str(class_id))
                            else:
                                class_name = names[int(class_id)]

                            detections.append(
                                {
                                    "Detected Condition": str(class_name),
                                    "Confidence (%)": round(float(conf) * 100, 2),
                                }
                            )

                    # ==================================================
                    # DETERMINE OVERALL DRONE CONDITION
                    # Damage has priority if ANY damage detection exists.
                    # ==================================================

                    if detections:
                        detection_df = (
                            pd.DataFrame(detections)
                            .sort_values("Confidence (%)", ascending=False)
                            .reset_index(drop=True)
                        )

                        normalized_conditions = [
                            str(condition)
                            .strip()
                            .lower()
                            .replace("_", " ")
                            .replace("-", " ")
                            for condition in detection_df["Detected Condition"]
                        ]

                        damage_rows = detection_df[
                            [
                                "damage" in condition or "damaged" in condition
                                for condition in normalized_conditions
                            ]
                        ]

                        healthy_rows = detection_df[
                            ["healthy" in condition for condition in normalized_conditions]
                        ]

                        if not damage_rows.empty:
                            display_confidence = float(
                                damage_rows["Confidence (%)"].max()
                            )

                            status_title = "⚠️ Damage Detected"
                            status_text = (
                                "The model detected a damage condition in the uploaded drone image."
                            )
                            status_type = "warning"

                        elif not healthy_rows.empty:
                            display_confidence = float(
                                healthy_rows["Confidence (%)"].max()
                            )

                            status_title = "✅ Healthy Drone"
                            status_text = (
                                "The model identified the drone as healthy based on the visible image features."
                            )
                            status_type = "success"

                        else:
                            display_confidence = float(
                                detection_df.iloc[0]["Confidence (%)"]
                            )

                            status_title = "🔎 Drone Condition Detected"
                            status_text = (
                                "The model detected the condition shown below."
                            )
                            status_type = "info"

                        # ==================================================
                        # RESULT HEADER
                        # ==================================================

                        if status_type == "success":
                            st.success(status_title)
                        elif status_type == "warning":
                            st.warning(status_title)
                        else:
                            st.info(status_title)

                        metric1, metric2 = st.columns(2)

                        with metric1:
                            st.metric(
                                "Confidence",
                                f"{display_confidence:.1f}%"
                            )

                        with metric2:
                            st.metric(
                                "Detections",
                                str(len(detection_df))
                            )

                        st.caption(status_text)

                        # ==================================================
                        # DETECTION DETAILS
                        # ==================================================

                        st.markdown("**Detection Details**")
                        st.dataframe(
                            detection_df,
                            width="stretch",
                            hide_index=True,
                        )

                        # ==================================================
                        # ANNOTATED IMAGE
                        # ==================================================

                        st.markdown("**Annotated Prediction**")

                        annotated_image = result.plot()

                        if annotated_image is not None:
                            annotated_rgb = annotated_image[:, :, ::-1]
                            st.image(
                                annotated_rgb,
                                width="stretch"
                            )

                        st.caption(
                            f"YOLO11n • Confidence threshold: {confidence_threshold:.2f} • "
                            f"Classes: Healthy Drone / Damage Drone"
                        )

                    else:
                        st.info("🔎 No confident detection")

                        st.caption(
                            f"The model did not identify Healthy Drone or Damage Drone "
                            f"above the selected confidence threshold of {confidence_threshold:.2f}."
                        )

                        st.markdown("**Uploaded Image**")
                        st.image(input_image, width="stretch")

                        st.caption(
                            "Try a slightly lower confidence threshold or upload a clearer drone image."
                        )

            except Exception as e:
                st.error(f"❌ Drone damage detection failed: {e}")


## 🤖 AI Assistant

if page == "🤖 AI Assistant":

    st.title("🤖 AI Assistant")

    st.caption(
        "Ask SmartLogix AI about orders, capacity and drone delivery"
    )


    st.subheader("💬 Ask the AI Agent")

    question = st.text_input(
        "📝 Enter your question",
        placeholder="Example: What is the status of ORD-005379?",
        key="ai_question"
    )

    ask_button = st.button(
        "🤖 Ask SmartLogix AI",
        type="primary"
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "⚠️ Please enter a question."
            )

        else:

            API_URL = os.getenv(
                "FASTAPI_URL",
                "http://127.0.0.1:8000"
            )

            try:

                with st.spinner(
                    "🤖 SmartLogix AI is processing your question..."
                ):

                    response = requests.post(
                        f"{API_URL}/ask-agent",
                        json={
                            "question": question
                        },
                        timeout=120
                    )

                if response.status_code == 200:

                    result = response.json()

                    answer = result.get(
                        "answer",
                        "No answer returned."
                    )

                    intent = result.get(
                        "intent",
                        "N/A"
                    )

                    selected_tool = result.get(
                        "selected_tool",
                        "N/A"
                    )

                    st.success(
                        "✅ AI response generated successfully"
                    )


                    st.subheader("💡 AI Answer")

                    st.info(answer)


                    st.subheader("🧠 Agent Details")

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "🎯 Detected Intent",
                            str(intent)
                        )

                    with col2:

                        st.metric(
                            "🔧 Selected Tool",
                            str(selected_tool)
                        )

                else:

                    st.error(
                        f"❌ FastAPI returned status code {response.status_code}"
                    )

                    st.code(
                        response.text
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the SmartLogix AI backend."
                )

                st.info(
                    "Make sure FastAPI is running on port 8000."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The AI Agent took too long to respond."
                )

            except Exception as e:

                st.error(
                    f"❌ Something went wrong: {e}"
                )
        
##  📦 Products

if page == "🛍️ Products":

    # ------------------------------------------------------------
    # CUSTOMER SHOPPING PAGE
    # ------------------------------------------------------------
    st.title("🛍️ SmartLogix Store")
    st.caption("Browse products, check ratings, and add your favourites to the cart.")

    # Cart is stored only for the current logged-in browser session.
    if "shopping_cart" not in st.session_state:
        st.session_state.shopping_cart = {}

    @st.cache_data(ttl=300)
    def load_customer_product_catalog():
        catalog_path = "cleaned_data/product_catalog_cleaned.csv"
        if not os.path.exists(catalog_path):
            return pd.DataFrame()
        return pd.read_csv(catalog_path)

    try:
        catalog = load_customer_product_catalog()
    except Exception as e:
        catalog = pd.DataFrame()
        st.error(f"Unable to load the product catalog: {e}")

    if catalog.empty:
        st.warning("⚠️ No products are available right now.")
        st.stop()

    # ------------------------------------------------------------
    # Normalize the catalog without changing the original dataset.
    # ------------------------------------------------------------
    catalog = catalog.copy()

    required_base_columns = [
        "product_id",
        "product_name",
        "category",
        "sub_category",
        "avg_rating",
        "stock_qty"
    ]

    for column in required_base_columns:
        if column not in catalog.columns:
            if column == "product_id":
                catalog[column] = [f"PRODUCT-{i + 1:05d}" for i in range(len(catalog))]
            elif column == "product_name":
                catalog[column] = "SmartLogix Product"
            elif column in ["category", "sub_category"]:
                catalog[column] = "General"
            elif column == "avg_rating":
                catalog[column] = 4.0
            elif column == "stock_qty":
                catalog[column] = 0

    catalog["product_id"] = catalog["product_id"].astype(str).str.strip()
    catalog["product_name"] = catalog["product_name"].astype(str).str.strip()
    catalog["category"] = catalog["category"].astype(str).str.strip()
    catalog["sub_category"] = catalog["sub_category"].astype(str).str.strip()
    catalog["avg_rating"] = pd.to_numeric(catalog["avg_rating"], errors="coerce").fillna(4.0).clip(0, 5)
    catalog["stock_qty"] = pd.to_numeric(catalog["stock_qty"], errors="coerce").fillna(0).clip(lower=0)

    # ------------------------------------------------------------
    # Detect a price column if the catalog has one.
    # Otherwise derive an approximate unit price from order history.
    # ------------------------------------------------------------
    price_candidates = [
        "price_inr",
        "unit_price_inr",
        "selling_price_inr",
        "price",
        "unit_price",
        "selling_price"
    ]

    price_column = next(
        (column for column in price_candidates if column in catalog.columns),
        None
    )

    if price_column is not None:
        catalog["display_price"] = pd.to_numeric(
            catalog[price_column], errors="coerce"
        )
    else:
        catalog["display_price"] = float("nan")

        if not orders_df.empty and "product_id" in orders_df.columns:
            order_price_data = orders_df.copy()
            if "order_value_inr" in order_price_data.columns:
                order_price_data["order_value_inr"] = pd.to_numeric(
                    order_price_data["order_value_inr"], errors="coerce"
                )
            else:
                order_price_data["order_value_inr"] = float("nan")

            if "quantity" in order_price_data.columns:
                order_price_data["quantity"] = pd.to_numeric(
                    order_price_data["quantity"], errors="coerce"
                )
            else:
                order_price_data["quantity"] = 1

            order_price_data = order_price_data[
                order_price_data["quantity"].fillna(0) > 0
            ].copy()

            if not order_price_data.empty:
                order_price_data["unit_price"] = (
                    order_price_data["order_value_inr"]
                    / order_price_data["quantity"]
                )

                price_lookup = (
                    order_price_data
                    .dropna(subset=["unit_price"])
                    .groupby(order_price_data["product_id"].astype(str).str.strip())["unit_price"]
                    .median()
                    .to_dict()
                )

                catalog["display_price"] = catalog["product_id"].map(price_lookup)

    catalog["display_price"] = pd.to_numeric(
        catalog["display_price"], errors="coerce"
    )

    # ------------------------------------------------------------
    # Product image support.
    # If the catalog later gets an image_url/image_path column,
    # the same page will automatically use it.
    # ------------------------------------------------------------
    image_candidates = [
        "image_url",
        "image",
        "image_path",
        "product_image",
        "product_image_url",
        "image_link",
        "thumbnail"
    ]

    image_column = next(
        (column for column in image_candidates if column in catalog.columns),
        None
    )

    def get_product_image(product_row):
        """Return a usable product image path/URL when the catalog provides one."""
        if image_column is None:
            return None

        image_value = product_row.get(image_column, None)
        if pd.isna(image_value):
            return None

        image_value = str(image_value).strip()
        if not image_value:
            return None

        if image_value.startswith("http://") or image_value.startswith("https://"):
            return image_value

        possible_paths = [
            image_value,
            os.path.join("cleaned_data", image_value),
            os.path.join("assets", image_value),
            os.path.join("images", image_value)
        ]

        for possible_path in possible_paths:
            if os.path.exists(possible_path):
                return possible_path

        return None

    # ------------------------------------------------------------
    # Attractive local fallback visual when no image exists.
    # This keeps the UI polished without depending on an external
    # image service.
    # ------------------------------------------------------------
    def create_product_visual(category, product_name):
        from PIL import ImageDraw, ImageFont

        width, height = 720, 430
        image = Image.new("RGB", (width, height), (242, 246, 250))
        draw = ImageDraw.Draw(image)

        category_text = str(category).strip().title()
        product_text = str(product_name).strip()

        category_icons = {
            "electronics": "💻",
            "mobile": "📱",
            "mobiles": "📱",
            "fashion": "👕",
            "clothing": "👕",
            "grocery": "🛒",
            "groceries": "🛒",
            "home": "🏠",
            "beauty": "✨",
            "sports": "⚽",
            "books": "📚",
            "toys": "🧸",
            "appliances": "🔌",
            "automotive": "🚗"
        }

        icon = "📦"
        category_lower = category_text.lower()
        for key, value in category_icons.items():
            if key in category_lower:
                icon = value
                break

        # Large simple product visual.
        draw.rounded_rectangle(
            (55, 45, width - 55, height - 55),
            radius=28,
            fill=(255, 255, 255),
            outline=(220, 226, 232),
            width=3
        )

        try:
            emoji_font = ImageFont.truetype("seguiemj.ttf", 120)
        except Exception:
            emoji_font = ImageFont.load_default()

        try:
            title_font = ImageFont.truetype("arial.ttf", 30)
        except Exception:
            title_font = ImageFont.load_default()

        try:
            category_font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            category_font = ImageFont.load_default()

        icon_box = draw.textbbox((0, 0), icon, font=emoji_font)
        icon_width = icon_box[2] - icon_box[0]
        icon_height = icon_box[3] - icon_box[1]
        draw.text(
            ((width - icon_width) / 2, 85),
            icon,
            font=emoji_font,
            fill=(45, 55, 65)
        )

        category_box = draw.textbbox((0, 0), category_text, font=category_font)
        category_width = category_box[2] - category_box[0]
        draw.text(
            ((width - category_width) / 2, 245),
            category_text,
            font=category_font,
            fill=(90, 100, 112)
        )

        short_name = product_text[:42]
        title_box = draw.textbbox((0, 0), short_name, font=title_font)
        title_width = title_box[2] - title_box[0]
        draw.text(
            ((width - title_width) / 2, 295),
            short_name,
            font=title_font,
            fill=(30, 35, 42)
        )

        return image

    # ------------------------------------------------------------
    # Header metrics
    # ------------------------------------------------------------
    available_products = int((catalog["stock_qty"] > 0).sum())
    total_catalog_products = int(catalog["product_id"].nunique())
    cart_count = sum(int(item["quantity"]) for item in st.session_state.shopping_cart.values())

    header_col1, header_col2, header_col3 = st.columns(3)

    with header_col1:
        st.metric("🛍️ Products", f"{total_catalog_products:,}")

    with header_col2:
        st.metric("📦 Available", f"{available_products:,}")

    with header_col3:
        st.metric("🛒 Cart Items", f"{cart_count:,}")

    st.markdown("---")

    # ------------------------------------------------------------
    # Search / category filters
    # ------------------------------------------------------------
    filter_col1, filter_col2 = st.columns([2.2, 1])

    with filter_col1:
        product_search = st.text_input(
            "🔎 Search products",
            placeholder="Search by product name or category...",
            key="customer_product_search"
        )

    with filter_col2:
        categories = ["All Categories"] + sorted(
            [
                str(value)
                for value in catalog["category"].dropna().unique()
                if str(value).strip()
            ]
        )

        selected_category = st.selectbox(
            "📂 Category",
            categories,
            key="customer_product_category"
        )

    filtered_products = catalog.copy()

    if product_search.strip():
        search_value = product_search.strip().lower()
        search_mask = (
            filtered_products["product_name"].str.lower().str.contains(search_value, na=False)
            | filtered_products["category"].str.lower().str.contains(search_value, na=False)
            | filtered_products["sub_category"].str.lower().str.contains(search_value, na=False)
        )
        filtered_products = filtered_products[search_mask]

    if selected_category != "All Categories":
        filtered_products = filtered_products[
            filtered_products["category"].eq(selected_category)
        ]

    # ------------------------------------------------------------
    # Cart summary / cart management
    # ------------------------------------------------------------
    with st.expander(
        f"🛒 My Cart ({cart_count} item{'s' if cart_count != 1 else ''})",
        expanded=cart_count > 0
    ):
        if not st.session_state.shopping_cart:
            st.info("Your cart is empty. Add products below to get started.")
        else:
            cart_total = 0.0
            cart_items_to_remove = []

            for product_id, cart_item in list(st.session_state.shopping_cart.items()):
                item_col1, item_col2, item_col3, item_col4 = st.columns([3, 1, 1, 1])

                item_price = float(cart_item.get("price", 0) or 0)
                item_quantity = int(cart_item.get("quantity", 1))
                item_total = item_price * item_quantity
                cart_total += item_total

                with item_col1:
                    st.markdown(
                        f"**{cart_item.get('name', 'Product')}**  \n"
                        f"{cart_item.get('category', 'Product')}"
                    )

                with item_col2:
                    new_quantity = st.number_input(
                        "Qty",
                        min_value=1,
                        max_value=max(1, int(cart_item.get("stock", 9999))),
                        value=item_quantity,
                        step=1,
                        key=f"cart_qty_{product_id}"
                    )

                    if new_quantity != item_quantity:
                        st.session_state.shopping_cart[product_id]["quantity"] = int(new_quantity)
                        st.rerun()

                with item_col3:
                    if item_price > 0:
                        st.write(f"₹{item_total:,.0f}")
                    else:
                        st.write("Price N/A")

                with item_col4:
                    if st.button("Remove", key=f"remove_cart_{product_id}"):
                        cart_items_to_remove.append(product_id)

            for product_id in cart_items_to_remove:
                st.session_state.shopping_cart.pop(product_id, None)
                st.rerun()

            st.markdown("---")
            total_col1, total_col2 = st.columns([3, 1])
            with total_col1:
                st.subheader("Cart Total")
            with total_col2:
                if cart_total > 0:
                    st.subheader(f"₹{cart_total:,.0f}")
                else:
                    st.subheader("Price N/A")

            st.success(
                "Your products are saved in the cart for this customer session. "
                "Order checkout can be connected to the Orders workflow next."
            )

    # ------------------------------------------------------------
    # Product cards
    # ------------------------------------------------------------
    st.subheader("✨ Shop Products")

    if filtered_products.empty:
        st.info("No products match your search.")
    else:
        # Show available products first.
        filtered_products = filtered_products.sort_values(
            by=["stock_qty", "avg_rating"],
            ascending=[False, False]
        ).reset_index(drop=True)

        cards_per_row = 3

        for start_index in range(0, len(filtered_products), cards_per_row):
            row = filtered_products.iloc[start_index:start_index + cards_per_row]
            card_columns = st.columns(cards_per_row)

            for card_position, (_, product_row) in enumerate(row.iterrows()):
                with card_columns[card_position]:
                    product_id = str(product_row["product_id"]).strip()
                    product_name = str(product_row["product_name"]).strip()
                    category = str(product_row["category"]).strip()
                    sub_category = str(product_row["sub_category"]).strip()
                    rating = float(product_row["avg_rating"])
                    stock = int(product_row["stock_qty"])

                    price_value = product_row.get("display_price", float("nan"))
                    has_price = pd.notna(price_value) and float(price_value) > 0
                    price_value = float(price_value) if has_price else 0.0

                    image_source = get_product_image(product_row)

                    if image_source is not None:
                        try:
                            st.image(image_source, width="stretch")
                        except Exception:
                            st.image(
                                create_product_visual(category, product_name),
                                width="stretch"
                            )
                    else:
                        st.image(
                            create_product_visual(category, product_name),
                            width="stretch"
                        )

                    st.markdown(f"### {product_name}")
                    st.caption(f"{category} • {sub_category}")

                    # Five-star rating display.
                    full_stars = int(rating)
                    half_star = (rating - full_stars) >= 0.5
                    empty_stars = max(0, 5 - full_stars - (1 if half_star else 0))
                    stars = "★" * full_stars
                    if half_star:
                        stars += "⯨"
                    stars += "☆" * empty_stars

                    st.markdown(
                        f"**{stars}**  `{rating:.1f}/5`"
                    )

                    if has_price:
                        st.markdown(f"## ₹{price_value:,.0f}")
                    else:
                        st.markdown("## Price on order")

                    if stock > 0:
                        st.success(f"In stock • {stock} available")
                    else:
                        st.error("Out of stock")

                    add_disabled = stock <= 0

                    if st.button(
                        "🛒 Add to Cart",
                        key=f"add_to_cart_{product_id}_{start_index}_{card_position}",
                        disabled=add_disabled,
                        use_container_width=True
                    ):
                        existing_item = st.session_state.shopping_cart.get(product_id)
                        existing_quantity = (
                            int(existing_item["quantity"])
                            if existing_item is not None
                            else 0
                        )

                        if existing_quantity >= stock:
                            st.warning("You have already added the maximum available quantity.")
                        else:
                            st.session_state.shopping_cart[product_id] = {
                                "name": product_name,
                                "category": category,
                                "price": price_value,
                                "quantity": existing_quantity + 1,
                                "stock": stock
                            }
                            st.success("Added to cart 🛒")
                            st.rerun()

                    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Similar-product recommendations
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Recommended for You")
    st.caption("Product recommendations are based on category, sub-category and tags using TF-IDF and cosine similarity.")

    try:
        products_for_recommendation, similarity_matrix = build_product_recommendation_model()

        if not products_for_recommendation.empty:
            recommendation_limit = min(5, len(products_for_recommendation))
            recommendation_rows = products_for_recommendation.head(recommendation_limit)

            recommendation_columns = st.columns(recommendation_limit)

            for rec_index, (_, recommendation_row) in enumerate(recommendation_rows.iterrows()):
                with recommendation_columns[rec_index]:
                    rec_name = str(recommendation_row.get("product_name", "Product"))
                    rec_category = str(recommendation_row.get("category", "General"))
                    rec_rating = pd.to_numeric(
                        recommendation_row.get("avg_rating", 4.0),
                        errors="coerce"
                    )
                    if pd.isna(rec_rating):
                        rec_rating = 4.0

                    st.image(
                        create_product_visual(rec_category, rec_name),
                        width="stretch"
                    )
                    st.markdown(f"**{rec_name}**")
                    st.caption(rec_category)
                    st.write(f"⭐ {float(rec_rating):.1f}/5")

    except Exception:
        st.info("Product recommendations will appear when the recommendation catalog is available.")

## ⭐ Reviews & Insights

elif page == "⭐ Reviews & Insights":

    st.header("💬 Reviews & Insights")
    st.caption(
        "Customer review sentiment analysis using TF-IDF and Logistic Regression."
    )

    # Load customer reviews dataset
    @st.cache_data
    def load_customer_reviews():
        try:
            return pd.read_csv(
                "cleaned_data/customer_reviews_cleaned.csv"
            )
        except Exception as e:
            st.error(f"Unable to load customer reviews: {e}")
            return pd.DataFrame()

    reviews = load_customer_reviews()

    if reviews.empty:
        st.warning("No customer review data available.")
        st.stop()

    st.success(
        f"🟢 Customer review data loaded successfully — {len(reviews):,} reviews"
    )

    # Prepare review text
    reviews["review_title"] = reviews["review_title"].fillna("").astype(str)
    reviews["review_text"] = reviews["review_text"].fillna("").astype(str)

    reviews["combined_text"] = (
        reviews["review_title"] + " " + reviews["review_text"]
    ).str.strip()

    # Train sentiment model
    @st.cache_resource
    def train_sentiment_model(review_text):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
            min_df=2
        )

        X = vectorizer.fit_transform(review_text)

        # Create sentiment labels from ratings
        return vectorizer, X

    # Create sentiment labels
    def rating_to_sentiment(rating):
        try:
            rating = float(rating)

            if rating <= 2:
                return "Negative"
            elif rating == 3:
                return "Neutral"
            else:
                return "Positive"

        except Exception:
            return "Neutral"

    if "rating" not in reviews.columns:
        st.error("The review dataset does not contain a rating column.")
        st.stop()

    reviews["sentiment"] = reviews["rating"].apply(rating_to_sentiment)

    # Train TF-IDF + Logistic Regression model
    @st.cache_resource
    def build_sentiment_model(texts, labels):

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
            min_df=2
        )

        X = vectorizer.fit_transform(texts)

        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        model.fit(X, labels)

        return vectorizer, model

    tfidf, sentiment_model = build_sentiment_model(
        reviews["combined_text"],
        reviews["sentiment"]
    )

    # Predict sentiment using the trained model
    X_reviews = tfidf.transform(reviews["combined_text"])
    reviews["predicted_sentiment"] = sentiment_model.predict(X_reviews)

    # KPI section
    total_reviews = len(reviews)
    positive_reviews = (
        reviews["predicted_sentiment"] == "Positive"
    ).sum()
    neutral_reviews = (
        reviews["predicted_sentiment"] == "Neutral"
    ).sum()
    negative_reviews = (
        reviews["predicted_sentiment"] == "Negative"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📝 Total Reviews",
            f"{total_reviews:,}"
        )

    with col2:
        st.metric(
            "😊 Positive",
            f"{positive_reviews:,}"
        )

    with col3:
        st.metric(
            "😐 Neutral",
            f"{neutral_reviews:,}"
        )

    with col4:
        st.metric(
            "😞 Negative",
            f"{negative_reviews:,}"
        )


    # Sentiment distribution
    st.subheader("📊 Sentiment Distribution")

    sentiment_counts = (
        reviews["predicted_sentiment"]
        .value_counts()
        .reindex(
            ["Positive", "Neutral", "Negative"],
            fill_value=0
        )
        .reset_index()
    )

    sentiment_counts.columns = ["Sentiment", "Reviews"]

    fig_sentiment = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Reviews",
        title="Customer Review Sentiment"
    )

    st.plotly_chart(
        fig_sentiment,
        width="stretch"
    )

    # Sentiment percentage
    st.subheader("📈 Sentiment Percentage")

    sentiment_percentage = (
        reviews["predicted_sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .reindex(
            ["Positive", "Neutral", "Negative"],
            fill_value=0
        )
        .reset_index()
    )

    sentiment_percentage.columns = [
        "Sentiment",
        "Percentage"
    ]

    sentiment_percentage["Percentage"] = (
        sentiment_percentage["Percentage"].round(2)
    )

    st.dataframe(
        sentiment_percentage,
        width="stretch",
        hide_index=True
    )


    # Sentiment by rating
    st.subheader("⭐ Sentiment by Rating")

    sentiment_by_rating = (
        reviews.groupby(
            ["rating", "predicted_sentiment"]
        )
        .size()
        .reset_index(name="Reviews")
    )

    fig_rating = px.bar(
        sentiment_by_rating,
        x="rating",
        y="Reviews",
        color="predicted_sentiment",
        barmode="group",
        title="Sentiment Distribution by Customer Rating"
    )

    st.plotly_chart(
        fig_rating,
        width="stretch"
    )


    # Sentiment by delivery mode
    if "delivery_mode" in reviews.columns:

        st.subheader("🚚 Sentiment by Delivery Mode")

        sentiment_delivery = (
            reviews.groupby(
                ["delivery_mode", "predicted_sentiment"]
            )
            .size()
            .reset_index(name="Reviews")
        )

        fig_delivery = px.bar(
            sentiment_delivery,
            x="delivery_mode",
            y="Reviews",
            color="predicted_sentiment",
            barmode="group",
            title="Customer Sentiment by Delivery Mode"
        )

        st.plotly_chart(
            fig_delivery,
            width="stretch"
        )


    # Product-level sentiment
    st.subheader("📦 Product-Level Sentiment")

    product_column = None

    for column in [
        "product_name",
        "product_id",
        "product"
    ]:
        if column in reviews.columns:
            product_column = column
            break

    if product_column is not None:

        product_sentiment = (
            reviews.groupby(
                [product_column, "predicted_sentiment"]
            )
            .size()
            .reset_index(name="Reviews")
        )

        fig_product = px.bar(
            product_sentiment,
            x=product_column,
            y="Reviews",
            color="predicted_sentiment",
            barmode="group",
            title="Product-Level Sentiment"
        )

        fig_product.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig_product,
            width="stretch"
        )

    else:
        st.info(
            "Product information is not available in the review dataset."
        )


    # Reliable products
    if product_column is not None:

        st.subheader("🏆 Reliable Products")

        product_review_counts = (
            reviews[product_column]
            .value_counts()
            .reset_index()
        )

        product_review_counts.columns = [
            product_column,
            "Review Count"
        ]

        reliable_products = product_review_counts[
            product_review_counts["Review Count"] >= 10
        ].copy()

        if not reliable_products.empty:

            reliable_sentiment = (
                reviews[
                    reviews[product_column].isin(
                        reliable_products[product_column]
                    )
                ]
                .groupby(product_column)
                .agg(
                    Review_Count=(
                        "predicted_sentiment",
                        "count"
                    ),
                    Positive_Count=(
                        "predicted_sentiment",
                        lambda x: (x == "Positive").sum()
                    ),
                    Negative_Count=(
                        "predicted_sentiment",
                        lambda x: (x == "Negative").sum()
                    )
                )
                .reset_index()
            )

            reliable_sentiment["Positive %"] = (
                reliable_sentiment["Positive_Count"]
                / reliable_sentiment["Review_Count"]
                * 100
            ).round(2)

            reliable_sentiment["Negative %"] = (
                reliable_sentiment["Negative_Count"]
                / reliable_sentiment["Review_Count"]
                * 100
            ).round(2)

            st.dataframe(
                reliable_sentiment.sort_values(
                    "Positive %",
                    ascending=False
                ),
                width="stretch",
                hide_index=True
            )


    # Review explorer
    st.subheader("🔎 Review Explorer")

    selected_sentiment = st.selectbox(
        "Filter by sentiment",
        [
            "All",
            "Positive",
            "Neutral",
            "Negative"
        ]
    )

    if selected_sentiment == "All":
        filtered_reviews = reviews.copy()
    else:
        filtered_reviews = reviews[
            reviews["predicted_sentiment"] == selected_sentiment
        ].copy()

    display_columns = []

    for column in [
        "review_title",
        "review_text",
        "rating",
        "delivery_mode",
        product_column
    ]:
        if column is not None and column in filtered_reviews.columns:
            display_columns.append(column)

    display_columns.append("predicted_sentiment")

    st.dataframe(
        filtered_reviews[display_columns].head(100),
        width="stretch",
        hide_index=True
    )


## 🔔 Notifications

elif page == "🔔 Notifications":

    st.header("🔔 Notifications")
    st.caption(
        "Operational alerts and AWS email notification services."
    )

    st.subheader("🚨 Operational Alerts")

    # Calculate current order statuses
    if orders_df.empty:
        st.warning("No order data is available.")
        st.stop()

    status_column = None

    for column in ["status", "order_status", "delivery_status"]:
        if column in orders_df.columns:
            status_column = column
            break

    if status_column is None:
        st.warning("Order status information is not available.")
        st.stop()

    status_values = (
        orders_df[status_column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    if "delivery_delay_hours" in orders_df.columns:
        delayed_orders = orders_df[
            pd.to_numeric(
                orders_df["delivery_delay_hours"],
                errors="coerce"
            ).gt(0)
        ]
    else:
        delayed_orders = orders_df.iloc[0:0].copy()

    transit_orders = orders_df[
        status_values.str.lower().eq("in transit")
    ]

    pending_orders = orders_df[
        status_values.str.lower().eq("pending")
    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🔴 Delayed Orders",
            f"{len(delayed_orders):,}"
        )

    with col2:
        st.metric(
            "🚚 In Transit",
            f"{len(transit_orders):,}"
        )

    with col3:
        st.metric(
            "⏳ Pending",
            f"{len(pending_orders):,}"
        )


    st.subheader("📋 Current Alerts")

    alerts = []

    if len(delayed_orders) > 0:
        alerts.append(
            {
                "Priority": "High",
                "Alert": "Delayed Orders",
                "Count": f"{len(delayed_orders):,}",
                "Description": (
                    f"{len(delayed_orders):,} order(s) "
                    "are currently delayed."
                )
            }
        )

    if len(transit_orders) > 0:
        alerts.append(
            {
                "Priority": "Medium",
                "Alert": "Orders In Transit",
                "Count": f"{len(transit_orders):,}",
                "Description": (
                    f"{len(transit_orders):,} order(s) "
                    "are currently in transit."
                )
            }
        )

    if len(pending_orders) > 0:
        alerts.append(
            {
                "Priority": "Medium",
                "Alert": "Pending Orders",
                "Count": f"{len(pending_orders):,}",
                "Description": (
                    f"{len(pending_orders):,} order(s) "
                    "are waiting for processing."
                )
            }
        )

    if alerts:

        alerts_df = pd.DataFrame(alerts)

        st.dataframe(
            alerts_df,
            width="stretch",
            hide_index=True
        )

    else:

        st.success(
            "✅ No operational alerts at the moment."
        )


    st.subheader("📧 Email Notification")

    st.write(
        "Send an operational notification using "
        "the AWS Lambda email notification service."
    )

    # Lambda email helper
    def send_lambda_email(subject, message):

        try:

            import boto3
            import json

            lambda_client = boto3.client(
                "lambda",
                region_name="ap-south-1"
            )

            payload = {
                "subject": subject,
                "message": message
            }

            response = lambda_client.invoke(
                FunctionName="smartlogix-email-notification",
                InvocationType="RequestResponse",
                Payload=json.dumps(payload)
            )

            response_payload = response.get(
                "Payload"
            )

            if response_payload:

                result = response_payload.read()

                if result:
                    return True, result.decode(
                        "utf-8",
                        errors="ignore"
                    )

            return True, "Email notification request sent."

        except Exception as e:

            return False, str(e)

    notification_subject = st.text_input(
        "Email Subject",
        value="SmartLogix Operational Notification"
    )

    notification_message = st.text_area(
        "Email Message",
        value=(
            "SmartLogix AI Operational Notification\n\n"
            "This is a test shipment notification from the SmartLogix AI system."
        ),
        height=150
    )

    if st.button(
        "📨 Send Email Notification",
        type="primary"
    ):

        if not notification_subject.strip():

            st.warning(
                "Please enter an email subject."
            )

        elif not notification_message.strip():

            st.warning(
                "Please enter an email message."
            )

        else:

            with st.spinner(
                "Sending email notification..."
            ):

                success, result = send_lambda_email(
                    notification_subject,
                    notification_message
                )

            if success:

                st.success(
                    "✅ Email notification sent successfully."
                )

            else:

                st.error(
                    f"❌ Unable to send email notification: {result}"
                )


    st.subheader("☁️ AWS Notification Architecture")

    st.info(
        "SmartLogix uses AWS Lambda to process email "
        "notifications and Amazon SES to deliver the "
        "notification email to the configured recipient."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🖥️ Streamlit")
        st.write(
            "Triggers the operational email notification."
        )

    with col2:
        st.markdown("### ⚡ AWS Lambda")
        st.write(
            "Processes the email notification request."
        )

    with col3:
        st.markdown("### 📧 Amazon SES")
        st.write(
            "Delivers the notification email."
        )
