# ===============================
# app.py — FINAL WORKING VERSION
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from google import genai
from pathlib import Path

# -------------------------------
# PATH SETUP
# -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"

DATA_FILE = DATA_PATH / "cleaned_superstore_data.parquet"


#-------------------------------
# Tiltle Config
#-------------------------------

st.set_page_config(
    page_title="Superstore Intelligence Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# DATA LOADING
# -------------------------------

@st.cache_data
def load_data():

    path = DATA_FILE

    st.write("📂 Loading data from:", path)

    if not path.exists():
        st.error(f"❌ Data file not found: {path}")
        st.stop()

    df = pd.read_parquet(path)

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df.dropna(subset=["Order Date"], inplace=True)

    return df


@st.cache_resource
def load_models():
    return {
        "v3_classifier": joblib.load(
            MODELS_PATH / "final_v3_profitability_classifier.joblib"
        ),
        "v3_forecaster": joblib.load(
            MODELS_PATH / "final_v3_profit_forecaster.joblib"
        ),
        "mba_model": joblib.load(
            MODELS_PATH / "final_predictive_mba_classifier.joblib"
        ),
    }


@st.cache_data
def create_feature_lookups(df):
    return {
        "postal_code_profitability": df.groupby("Postal Code")[
            "Profit_Margin"
        ].mean().to_dict(),
        "state_sales_volume": df.groupby("State")["Total Sales"].sum().to_dict(),
        "subcategory_avg_margin": df.groupby("Sub-Category")[
            "Profit_Margin"
        ].mean().to_dict(),
        "category_avg_discount": df.groupby("Category")["Discount"]
        .mean()
        .to_dict(),
        "customer_avg_order_size": df.groupby("Customer ID")[
            "Total Sales"
        ].mean().to_dict(),
        "product_features": df.groupby("Product Name")
        .agg(
            Category=("Category", "first"),
            SubCategory=("Sub-Category", "first"),
            AvgSalesPrice=("Sales Price", "mean"),
        )
        .reset_index(),
    }


# -------------------------------
# GEMINI CONFIG (google-genai)
# -------------------------------

try:
    GEMINI_CLIENT = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    GEMINI_CLIENT = None

def create_ceo_memo_prompt():
    return """
You are a Chief Data Scientist reporting to the CEO.

Draft a strategic memo with:

1. Executive Summary
2. Customer Segments
3. Profit Drivers
4. Recommendations
5. Conclusion
"""


# -------------------------------
# MODEL FEATURES (LOCKED)   
# -------------------------------
v3_numerical_features = [
    "Sales Price",
    "Quantity",
    "Discount",
    "month_of_year",
    "days_since_first_order",
    "postal_code_profitability",
    "state_sales_volume",
    "subcategory_avg_margin",
    "category_avg_discount",
    "customer_avg_order_size",
]

v3_categorical_features = [
    "Segment",
    "Region",
    "State",
    "Category",
    "Sub-Category",
]

def prepare_v3_features(df):
    return df[v3_numerical_features + v3_categorical_features]


# --- Persona Content (FULL VERSION) ---
PERSONAS = {
    "🏆 Super Loyalists (Top 33%)": {
        "stats": {"Recency": "116.0 days", "Frequency": "8.4 orders", "Monetary Value": "$6,585.2"},
        "narrative": """
### 1. Persona Title
🏆 **"The Established Loyalist" (Rachel)**

### 2. Narrative Summary
Meet Rachel. She represents the bedrock of our customer base—a long-standing, high-value patron who has consistently chosen our brand for significant purchases over the years. As a busy professional in her late 30s or 40s with high disposable income, Rachel values quality, durability, and a brand she can trust. While her deep loyalty is proven by her purchase history, it has been nearly four months since her last order, indicating she may be between buying cycles. Her engagement is crucial for our long-term success.

### 3. Key Attributes
- **High Lifetime Value:** Total spend over $6,500.
- **Brand Loyal & Repeat Purchaser:** An average of 8.4 orders demonstrates a clear preference for our brand.
- **High Average Order Value (AOV):** Her average order is substantial (approx. $784), indicating a focus on premium items.
- **Infrequent Purchaser:** A recency of 116 days suggests a long buying cycle, possibly due to the nature of the products purchased.

### 4. Motivations & Needs
- **Seeks Quality and Durability:** Motivated by long-term value, not short-term deals.
- **Needs Trust and Reliability:** Requires deep trust in our product, service, and delivery.
- **Appreciates Recognition:** Expects a level of service that matches her investment in the brand.
- **Values a Seamless Experience:** Needs a frictionless purchasing process.

### 5. Actionable Marketing Strategies
- **Personalized Re-engagement:** Send targeted emails highlighting "What's New Since Your Last Visit," featuring products complementary to her past purchases.
- **Implement a VIP/Loyalty Tier:** Automatically enroll this segment into a top tier with exclusive benefits like early access to new collections or a dedicated customer service line.
- **High-Touch Content Marketing:** Send exclusive content that reinforces their smart purchase decisions, like care guides or behind-the-scenes looks at craftsmanship.
- **Solicit High-Value Feedback:** Invite her to a "customer advisory panel" to get feedback on potential new products, reinforcing her importance.
        """
    },
    "💰 High Spenders (Middle 33%)": {
        "stats": {"Recency": "134.1 days", "Frequency": "5.8 orders", "Monetary Value": "$3,334.4"},
        "narrative": """
### 1. Persona Title
💰 **"The Selective Investor" (Mark)**

### 2. Narrative Summary
Meet Mark. He is a discerning customer who invests significantly when he has a specific need. Representing our middle-tier, Mark is willing to spend over $3,000, but his engagement is less frequent. His purchasing cycle of over four months suggests he is a project-based or event-driven buyer. While he contributes major revenue, his loyalty is not guaranteed. The key challenge is to bridge the long gaps between his purchases and build a more consistent relationship.

### 3. Key Attributes
- **Significant Lifetime Value (LTV):** Total spend exceeds $3,300, making a substantial impact on revenue.
- **High Average Order Value (AOV):** Spends approximately $575 per order, showing a preference for high-quality items.
- **Moderately Frequent Purchaser:** Has made several repeat purchases (5.8 orders) but is not a deeply embedded loyalist.
- **Currently Dormant:** A 134-day recency is a critical flag; he is at risk of being acquired by a competitor.

### 4. Motivations & Needs
- **Solution-Oriented:** Shops to solve a specific, often large-scale problem (e.g., furnishing a room, upgrading tech).
- **Driven by Research and Value:** Needs strong social proof, detailed product specs, and clear value propositions.
- **Needs a Compelling Reason to Return:** His purchasing habit is not automatic and needs to be prompted.
- **Trust in Quality:** Willing to spend more for products that will last.

### 5. Actionable Marketing Strategies
- **"Next Project" Themed Campaigns:** Create content that inspires their next purchase (e.g., if they bought a desk, target them with content about "Building the Perfect Home Office").
- **Strategic "Bounce-Back" Offer:** After a large purchase, send a time-sensitive, high-value offer to incentivize a follow-up purchase and shorten the recency period.
- **Promote Cross-Category Discovery:** Use targeted marketing to introduce him to other relevant product categories.
- **Re-engage with High-Impact Product Launches:** Use this segment as a primary audience for announcing major new products, framing them as a worthy investment.
        """
    },
    "😴 Dormant Customers (Bottom 33%)": {
        "stats": {"Recency": "234.8 days", "Frequency": "3.4 orders", "Monetary Value": "$1,676.9"},
        "narrative": """
### 1. Persona Title
😴 **"The Forgotten Patron" (Priya)**

### 2. Narrative Summary
Meet Priya. She represents a significant and concerning segment—a once-promising patron who has since gone silent. Priya initially made 3-4 purchases, spending over $1,600, but it's now been over seven months since her last interaction. For all intents and purposes, she has churned. Re-engaging Priya is a high-effort, high-reward challenge, but winning her back requires a powerful, strategic intervention.

### 3. Key Attributes
- **Deeply Lapsed/Churned:** An average recency of 235 days means this segment is no longer actively considering our brand.
- **Proven Historical Value:** A lifetime spend of nearly $1,700 makes them too valuable to ignore completely.
- **Past Multi-Purchaser:** An average of 3.4 orders shows they were once satisfied enough to return.
- **High Average Order Value (AOV):** Their average order value was substantial (approx. $493).

### 4. Motivations & Needs
- **Needs a Strong Reason to Return:** Standard marketing is insufficient. They require a compelling incentive.
- **Lack of Top-of-Mind Awareness:** Our brand has fallen off their radar.
- **Potentially Dissatisfied or Indifferent:** Their dormancy could stem from a negative experience or a competitor meeting their needs better.
- **Requires Re-Proof of Value:** We need to prove our brand is still the best choice through innovation or a superior value proposition.

### 5. Actionable Marketing Strategies
- **Execute a High-Impact Win-Back Campaign:** Deploy a multi-channel campaign with a steep offer (e.g., "25% off to welcome you back") and messaging that explicitly acknowledges their absence.
- **Launch a Feedback-Oriented Survey:** Instead of a sales pitch, send an email asking, "Where did we go wrong?" with a strong incentive for completion. This provides invaluable churn data.
- **"Last Chance" Drip Campaign:** For non-responders, an automated series highlighting major brand improvements since their last visit can help rebuild trust.
- **Isolate and Suppress:** If a user remains unengaged, suppress them from costly marketing to avoid wasting spend.
        """
    }
}


# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

def create_live_features(df_input, lookups, base_df):
    df = df_input.copy()

    df["month_of_year"] = df["Order Date"].dt.month
    customer_id = df["Customer ID"].iloc[0]

    if customer_id == "New Customer":
        df["days_since_first_order"] = 0
        df["customer_avg_order_size"] = base_df["Total Sales"].mean()
    else:
        first_order_date = base_df[
            base_df["Customer ID"] == customer_id
        ]["Order Date"].min()

        if pd.notna(first_order_date):
            df["days_since_first_order"] = (
                df["Order Date"] - first_order_date
            ).dt.days.iloc[0]
        else:
            df["days_since_first_order"] = 0

        df["customer_avg_order_size"] = lookups[
            "customer_avg_order_size"
        ].get(customer_id, base_df["Total Sales"].mean())

    df["postal_code_profitability"] = lookups[
        "postal_code_profitability"
    ].get(df["Postal Code"].iloc[0], base_df["Profit_Margin"].mean())

    df["state_sales_volume"] = lookups["state_sales_volume"].get(
        df["State"].iloc[0], base_df["Total Sales"].mean()
    )

    df["subcategory_avg_margin"] = lookups[
        "subcategory_avg_margin"
    ].get(df["Sub-Category"].iloc[0], base_df["Profit_Margin"].mean())

    df["category_avg_discount"] = lookups[
        "category_avg_discount"
    ].get(df["Category"].iloc[0], base_df["Discount"].mean())

    return df


# -------------------------------
# STRATEGIC PAGE
# -------------------------------
def render_customer_segmentation_page():
    st.title("Customer Segment Explorer")
    st.markdown("Explore AI-generated personas for customer segments discovered by our advanced deep learning model.")
    
    selected_persona = st.selectbox("Select a persona:", list(PERSONAS.keys()))
    if selected_persona:
        p_data = PERSONAS[selected_persona]
        c1, c2, c3 = st.columns(3); c1.metric("Avg. Recency", p_data["stats"]["Recency"]); c2.metric("Avg. Frequency", p_data["stats"]["Frequency"]); c3.metric("Avg. Monetary Value", p_data["stats"]["Monetary Value"])
        st.markdown("---"); st.markdown(p_data["narrative"])

def render_home_page(df):
    st.title("Superstore Intelligence Engine")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", f"${df['Total Sales'].sum():,.0f}")
    c2.metric("Total Profit", f"${df['Total Profit'].sum():,.0f}")
    c3.metric("Unique Customers", df["Customer ID"].nunique())


def render_prediction_suite_page(df, models, lookups):
    st.title("Prediction Suite")

    c1, c2 = st.columns(2)

    with c1:
        segment = st.selectbox("Segment", df["Segment"].unique())
        region = st.selectbox("Region", df["Region"].unique())
        state = st.selectbox("State", df[df["Region"] == region]["State"].unique())
        category = st.selectbox("Category", df["Category"].unique())
        sub_category = st.selectbox(
            "Sub-Category", df[df["Category"] == category]["Sub-Category"].unique()
        )

    with c2:
        price_stats = df[df["Sub-Category"] == sub_category]["Sales Price"].describe()
        sales_price = st.number_input(
            "Sales Price",
            float(price_stats["min"]),
            float(price_stats["max"]),
            float(price_stats["mean"]),
            step=0.01,
        )
        quantity = st.number_input("Quantity", min_value=1, value=2)
        discount = st.slider("Discount (%)", 0, 100, 10) / 100
        postal_code = st.selectbox(
            "Postal Code", df[df["State"] == state]["Postal Code"].unique()
        )
        customer_id = st.selectbox(
            "Customer ID", ["New Customer"] + sorted(df["Customer ID"].unique())
        )

    if st.button("Analyze Profitability"):
        raw_input = {
            "Segment": segment,
            "Region": region,
            "State": state,
            "Category": category,
            "Sub-Category": sub_category,
            "Sales Price": sales_price,
            "Quantity": quantity,
            "Discount": discount,
            "Postal Code": postal_code,
            "Customer ID": customer_id,
            "Order Date": pd.Timestamp.now(),
        }

        df_live = create_live_features(pd.DataFrame([raw_input]), lookups, df)
        X = prepare_v3_features(df_live)

        clf = models["v3_classifier"]
        reg = models["v3_forecaster"]

        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]

        if pred == 1:
            st.success(f"Profitable (Confidence: {proba[1]:.2%})")
            profit = np.expm1(reg.predict(X)[0])
            st.info(f"Predicted Profit: ${profit:.2f}")
        else:
            st.error(f"Not Profitable (Confidence: {proba[0]:.2%})")

def render_recommender_page(df, models, lookups):
    st.title("Predictive Market Basket")

    products = sorted(df["Product Name"].unique())
    a = st.selectbox("Product A", products)
    b = st.selectbox("Product B", products, index=1 if len(products) > 1 else 0)

    if st.button("Predict Co-Purchase"):
        feats = lookups["product_features"]
        fa = feats[feats["Product Name"] == a].iloc[0]
        fb = feats[feats["Product Name"] == b].iloc[0]

        X = pd.DataFrame([{
            "A_Category": fa["Category"],
            "A_SubCategory": fa["SubCategory"],
            "A_AvgSalesPrice": fa["AvgSalesPrice"],
            "B_Category": fb["Category"],
            "B_SubCategory": fb["SubCategory"],
            "B_AvgSalesPrice": fb["AvgSalesPrice"],
        }])

        proba = models["mba_model"].predict_proba(X)[0][1]
        st.info(f"Co-purchase probability: {proba:.2%}")

def render_strategic_insights_page():
    st.title("Strategic Insights Dashboard")

    st.subheader("Key Drivers of Profit (SHAP Analysis)")

    c1, c2 = st.columns(2)

    with c1:
        path = REPORTS_PATH / "final_shap_plots" / "shap_summary_bar.png"
        if path.exists():
            st.image(str(path))
        else:
            st.error("Bar plot image not found.")

    with c2:
        path = REPORTS_PATH / "final_shap_plots" / "shap_summary_beeswarm.png"
        if path.exists():
            st.image(str(path))
        else:
            st.error("Beeswarm plot image not found.")

    st.markdown("---")

    st.subheader("AI Strategy Co-Pilot")

    if st.button("Generate AI Strategy Memo"):

        if not GEMINI_CLIENT:
            st.warning("Gemini disabled.")
            return

        model_candidates = [
            "models/gemini-pro",
            "models/gemini-1.0-pro",
            "models/gemma-3-27b-it",
        ]

        import datetime
        timestamp = datetime.datetime.now().isoformat()

        prompt = create_ceo_memo_prompt() + f"""

Generate a fresh version of this memo.
Current time: {timestamp}
"""

        for model_name in model_candidates:
            try:
                response = GEMINI_CLIENT.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )

                st.caption(f"Using model: {model_name}")
                st.markdown(response.text)
                break

            except Exception:
                continue

        else:
            st.error("No compatible Gemini model available for this API key or region.")







def create_ceo_memo_prompt():
    return """
You are a Chief Data Scientist reporting to the CEO.

Draft a strategic memo with:

1. Executive Summary
2. Customer Segments
3. Profit Drivers
4. Recommendations
5. Conclusion
"""


# -------------------------------
# MAIN
# -------------------------------
def main():

    base_df = load_data()

    base_df["Profit_Margin"] = np.where(
        base_df["Total Sales"] > 0,
        base_df["Total Profit"] / base_df["Total Sales"],
        0,
    )

    models = load_models()
    lookups = create_feature_lookups(base_df)

    # -------------------------------
    # SIDEBAR NAVIGATION (INSIDE MAIN)
    # -------------------------------

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Customer Segmentation",
            "Prediction Suite",
            "Product Recommender",
            "Strategic Insights",
        ],
    )

    if page == "Home":
        render_home_page(base_df)

    elif page == "Customer Segmentation":
        render_customer_segmentation_page()

    elif page == "Prediction Suite":
        render_prediction_suite_page(base_df, models, lookups)

    elif page == "Product Recommender":
        render_recommender_page(base_df, models, lookups)

    elif page == "Strategic Insights":
        render_strategic_insights_page()



    # selection = st.sidebar.radio("Go to", list(pages.keys()))
    # pages[selection]()


if __name__ == "__main__":
    main()
